#include "mragapp.h"
#include "config.h"
#include "chunk.h"
#include "document.h"
#include "vectordatabase.h"
#include "llamamodel.h"
#include "embeddingengine.h"
#include "generationengine.h"

#include "llama.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#endif

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <cassert>
#include <iterator>
#include <algorithm>
#include <cctype>
#include <cstdlib>

static std::string gExecutablePath;

// RAII 管理 llama backend 生命周期：构造时 init，析构时 free
struct BackendGuard {
    BackendGuard() { llama_backend_init(); }
    ~BackendGuard() { llama_backend_free(); }
    BackendGuard(const BackendGuard &) = delete;
    BackendGuard &operator=(const BackendGuard &) = delete;
};

static void nullLogCallback(ggml_log_level /*level*/, const char * /*text*/, void * /*user_data*/) {
}

static void buildKnowledgeBase(const std::vector<std::string> &txtPaths,
                                const std::string &dbPath,
                                const AppConfig &config);

// 核心问答流程：query Embedding → DB 检索 → 显示片段预览 → 生成回答 → 输出
static void doQuery(VectorDatabase &db, EmbeddingEngine &embEngine, GenerationEngine &genEngine,
                    const std::string &query, const AppConfig &config);
static bool sameRuntimeModels(const AppConfig &lhs, const AppConfig &rhs);
static bool reloadFromConfig(AppConfig &activeConfig,
                             VectorDatabase &db,
                             std::unique_ptr<EmbeddingEngine> &embEngine,
                             std::unique_ptr<GenerationEngine> &genEngine);

static std::string trimCommandInput(const std::string &input) {
    size_t begin = 0;
    while (begin < input.size() &&
           std::isspace(static_cast<unsigned char>(input[begin]))) {
        ++begin;
    }
    if (begin + 3 <= input.size() &&
        static_cast<unsigned char>(input[begin]) == 0xEF &&
        static_cast<unsigned char>(input[begin + 1]) == 0xBB &&
        static_cast<unsigned char>(input[begin + 2]) == 0xBF) {
        begin += 3;
    }
    size_t end = input.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(input[end - 1]))) {
        --end;
    }
    return input.substr(begin, end - begin);
}

static bool containsIgnoreCase(const std::string &text, const std::string &needle) {
    auto toLower = [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    };
    std::string lowerText(text.size(), '\0');
    std::string lowerNeedle(needle.size(), '\0');
    std::transform(text.begin(), text.end(), lowerText.begin(), toLower);
    std::transform(needle.begin(), needle.end(), lowerNeedle.begin(), toLower);
    return lowerText.find(lowerNeedle) != std::string::npos;
}

static std::string utf8Preview(const std::string &text, size_t maxChars) {
    size_t pos = 0;
    size_t count = 0;
    while (pos < text.size() && count < maxChars) {
        unsigned char c = static_cast<unsigned char>(text[pos]);
        size_t len = 1;
        if (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        if (pos + len > text.size()) break;
        pos += len;
        ++count;
    }
    return text.substr(0, pos);
}

static std::string modelPathFromName(const std::string &modelsDir, const std::string &fileName) {
    return modelsDir + "/" + fileName;
}

static bool samePathText(const std::string &lhs, const std::string &rhs) {
    auto normalize = [](std::string path) {
        std::replace(path.begin(), path.end(), '\\', '/');
        if (path.rfind("./", 0) == 0) path.erase(0, 2);
        return path;
    };
    return normalize(lhs) == normalize(rhs);
}

static std::string quoteArg(const std::string &arg) {
    std::string quoted = "\"";
    for (char ch : arg) {
        if (ch == '"') quoted += "\\\"";
        else quoted += ch;
    }
    quoted += "\"";
    return quoted;
}

static bool probeEmbeddingModel(const std::string &modelPath) {
    if (gExecutablePath.empty()) return true;
#ifdef _WIN32
    std::string command = quoteArg(gExecutablePath) + " --probe-emb " + quoteArg(modelPath) + " >nul 2>nul";
#else
    std::string command = quoteArg(gExecutablePath) + " --probe-emb " + quoteArg(modelPath) + " >/dev/null 2>&1";
#endif
    int rc = std::system(command.c_str());
    return rc == 0;
}

static void rebuildEmbeddings(VectorDatabase &db, EmbeddingEngine &engine) {
    std::vector<std::vector<float>> rebuilt;
    rebuilt.reserve(db.size());
    const auto &chunks = db.chunks();
    for (size_t i = 0; i < chunks.size(); ++i) {
        std::vector<float> embedding = engine.generateEmbedding(chunks[i].text);
        if (embedding.empty()) {
            throw std::runtime_error("重新生成片段向量失败");
        }
        rebuilt.push_back(std::move(embedding));
        if ((i + 1) % 10 == 0 || i + 1 == chunks.size()) {
            std::cout << "\r[INFO] 重建向量进度: " << (i + 1)
                      << "/" << chunks.size() << std::flush;
        }
    }
    std::cout << std::endl;

    auto &mutableChunks = db.mutableChunks();
    for (size_t i = 0; i < mutableChunks.size(); ++i) {
        mutableChunks[i].embedding = std::move(rebuilt[i]);
    }
}

// 交互式热切换：扫描 models/ 目录，分别选择生成模型和 Embedding 模型
static bool interactiveReload(AppConfig &activeConfig,
                               VectorDatabase &db,
                               std::unique_ptr<EmbeddingEngine> &embEngine,
                               std::unique_ptr<GenerationEngine> &genEngine) {
    std::vector<std::string> ggufFiles;
    const std::string modelsDir = "./models";

    // 扫描 models/ 目录下所有 .gguf 文件
    {
#ifdef _WIN32
        std::string pattern = modelsDir + "\\*.gguf";
        WIN32_FIND_DATAA fd;
        HANDLE hFind = FindFirstFileA(pattern.c_str(), &fd);
        if (hFind != INVALID_HANDLE_VALUE) {
            do { ggufFiles.push_back(fd.cFileName); } while (FindNextFileA(hFind, &fd));
            FindClose(hFind);
        }
#else
        DIR *dir = opendir(modelsDir.c_str());
        if (dir) {
            struct dirent *entry;
            while ((entry = readdir(dir)) != nullptr) {
                std::string name = entry->d_name;
                if (name.size() >= 5 && name.compare(name.size() - 5, 5, ".gguf") == 0)
                    ggufFiles.push_back(name);
            }
            closedir(dir);
        }
#endif
    }

    if (ggufFiles.empty()) {
        std::cerr << "[ERROR] models/ 目录中未找到 .gguf 文件" << std::endl;
        return false;
    }

    std::vector<std::string> genFiles;
    std::vector<std::string> embFiles;
    for (const auto &file : ggufFiles) {
        if (containsIgnoreCase(file, "bge") || containsIgnoreCase(file, "embed")) {
            embFiles.push_back(file);
        } else {
            genFiles.push_back(file);
        }
    }
    if (genFiles.empty()) genFiles = ggufFiles;
    if (embFiles.empty()) embFiles = ggufFiles;

    std::cout << "\n--- 可用的生成模型 ---" << std::endl;
    for (size_t i = 0; i < genFiles.size(); ++i) {
        std::cout << "  [" << (i + 1) << "] " << genFiles[i] << std::endl;
    }
    std::cout << "------------------------" << std::endl;

    // 选择生成模型
    std::cout << "请选择生成模型编号（回车跳过不改）: " << std::flush;
    std::string input;
    std::getline(std::cin, input);

    size_t idx = 0;
    bool changed = false;

    if (!input.empty()) {
        try { idx = static_cast<size_t>(std::stoul(input)); } catch (...) { idx = 0; }
        if (idx >= 1 && idx <= genFiles.size()) {
            std::string newGenPath = modelPathFromName(modelsDir, genFiles[idx - 1]);
            if (samePathText(newGenPath, activeConfig.model.gen_model_path)) {
                std::cout << "[INFO] 生成模型未变化，跳过加载" << std::endl;
            } else {
                const std::string oldEmbPath = activeConfig.model.emb_model_path;
                embEngine.reset();
                genEngine.reset();
                std::cout << "[INFO] 正在切换到生成模型: " << genFiles[idx - 1] << std::endl;
                try {
                    genEngine = std::make_unique<GenerationEngine>(
                        newGenPath, activeConfig.generation, activeConfig.embedding,
                        activeConfig.model.n_gpu_layers);
                    embEngine = std::make_unique<EmbeddingEngine>(
                        oldEmbPath, activeConfig.embedding, activeConfig.generation,
                        activeConfig.model.n_gpu_layers);
                } catch (...) {
                    genEngine.reset();
                    embEngine.reset();
                    throw;
                }
                activeConfig.model.gen_model_path = newGenPath;
                changed = true;
            }
        } else {
            std::cout << "[INFO] 无效编号，跳过生成模型切换" << std::endl;
        }
    }

    std::cout << "\n--- 可用的 Embedding 模型 ---" << std::endl;
    for (size_t i = 0; i < embFiles.size(); ++i) {
        std::cout << "  [" << (i + 1) << "] " << embFiles[i] << std::endl;
    }
    std::cout << "-----------------------------" << std::endl;
    std::cout << "请选择 Embedding 模型编号（回车跳过不改）: " << std::flush;
    std::getline(std::cin, input);

    if (!input.empty()) {
        try { idx = static_cast<size_t>(std::stoul(input)); } catch (...) { idx = 0; }
        if (idx >= 1 && idx <= embFiles.size()) {
            std::string newEmbPath = modelPathFromName(modelsDir, embFiles[idx - 1]);
            if (samePathText(newEmbPath, activeConfig.model.emb_model_path)) {
                std::cout << "[INFO] Embedding 模型未变化，跳过加载" << std::endl;
            } else {
                std::cout << "[INFO] 正在探测 Embedding 模型..." << std::endl;
                if (!probeEmbeddingModel(newEmbPath)) {
                    throw std::runtime_error("目标文件不能作为 Embedding 模型加载");
                }
                std::cout << "[INFO] 正在加载 Embedding 模型并重建当前数据库向量..." << std::endl;
                auto newEngine = std::make_unique<EmbeddingEngine>(
                    newEmbPath, activeConfig.embedding, activeConfig.generation,
                    activeConfig.model.n_gpu_layers);
                rebuildEmbeddings(db, *newEngine);
                embEngine = std::move(newEngine);
                activeConfig.model.emb_model_path = newEmbPath;
                changed = true;
            }
        } else {
            std::cout << "[INFO] 无效编号，跳过 Embedding 模型切换" << std::endl;
        }
    }

    if (changed) {
        std::cout << "[INFO] 正在写回模型选择..." << std::endl;
        if (saveConfig("config.json", activeConfig)) {
            std::cout << "[INFO] 已将当前模型选择写回 config.json" << std::endl;
        }
        std::cout << "[INFO] 模型热切换完成" << std::endl;
    } else {
        std::cout << "[INFO] 未变更任何模型" << std::endl;
    }
    return changed;
}

static bool sameRuntimeModels(const AppConfig &lhs, const AppConfig &rhs) {
    return lhs.model.emb_model_path == rhs.model.emb_model_path
        && lhs.model.gen_model_path == rhs.model.gen_model_path
        && lhs.model.n_gpu_layers == rhs.model.n_gpu_layers
        && lhs.embedding.n_ctx == rhs.embedding.n_ctx
        && lhs.embedding.n_batch == rhs.embedding.n_batch
        && lhs.embedding.n_ubatch == rhs.embedding.n_ubatch
        && lhs.generation.n_ctx == rhs.generation.n_ctx
        && lhs.generation.n_batch == rhs.generation.n_batch
        && lhs.generation.n_ubatch == rhs.generation.n_ubatch
        && lhs.generation.max_output_tokens == rhs.generation.max_output_tokens
        && lhs.generation.temperature == rhs.generation.temperature
        && lhs.generation.top_k == rhs.generation.top_k
        && lhs.generation.top_p == rhs.generation.top_p;
}

static bool reloadFromConfig(AppConfig &activeConfig,
                             VectorDatabase &db,
                             std::unique_ptr<EmbeddingEngine> &embEngine,
                             std::unique_ptr<GenerationEngine> &genEngine) {
    AppConfig latestConfig = loadConfig("config.json");
    if (!sameRuntimeModels(activeConfig, latestConfig)) {
        std::cout << "[INFO] 检测到 config.json 变化，正在热切换模型/推理参数..." << std::endl;
        if (!samePathText(activeConfig.model.gen_model_path, latestConfig.model.gen_model_path) ||
            activeConfig.generation.n_ctx != latestConfig.generation.n_ctx ||
            activeConfig.generation.n_batch != latestConfig.generation.n_batch ||
            activeConfig.generation.n_ubatch != latestConfig.generation.n_ubatch) {
            genEngine.reset();
            genEngine = std::make_unique<GenerationEngine>(latestConfig.model.gen_model_path,
                                                           latestConfig.generation, latestConfig.embedding,
                                                           latestConfig.model.n_gpu_layers);
        }
        if (!samePathText(activeConfig.model.emb_model_path, latestConfig.model.emb_model_path) ||
            activeConfig.embedding.n_ctx != latestConfig.embedding.n_ctx ||
            activeConfig.embedding.n_batch != latestConfig.embedding.n_batch ||
            activeConfig.embedding.n_ubatch != latestConfig.embedding.n_ubatch) {
            auto newEngine = std::make_unique<EmbeddingEngine>(
                latestConfig.model.emb_model_path, latestConfig.embedding,
                latestConfig.generation, latestConfig.model.n_gpu_layers);
            rebuildEmbeddings(db, *newEngine);
            embEngine = std::move(newEngine);
        }
        activeConfig = latestConfig;
        std::cout << "[INFO] 配置热切换完成" << std::endl;
        return true;
    }

    activeConfig.retrieval = latestConfig.retrieval;
    return false;
}

// 离线建库全流程：逐个 TXT 文档处理 → 合并所有 Chunk → 逐一生成 Embedding → 存库
static void buildKnowledgeBase(const std::vector<std::string> &txtPaths,
                                const std::string &dbPath,
                                const AppConfig &config) {
    assert(!txtPaths.empty());
    assert(!dbPath.empty());

    DocumentProcessor processor(config.document);
    std::vector<Chunk> chunks;
    for (const auto &txtPath : txtPaths) {
        assert(!txtPath.empty());
        std::cout << "[INFO] 正在处理文档: " << txtPath << std::endl;
        std::vector<Chunk> docChunks = processor.processNovel(txtPath);
        std::cout << "[INFO] 当前文档切分完成，共 " << docChunks.size() << " 个片段" << std::endl;
        chunks.insert(chunks.end(),
                      std::make_move_iterator(docChunks.begin()),
                      std::make_move_iterator(docChunks.end()));
    }
    std::cout << "[INFO] 全部文档切分完成，共 " << chunks.size() << " 个片段" << std::endl;

    if (chunks.empty()) {
        std::cerr << "[ERROR] 未生成任何片段，建库终止" << std::endl;
        return;
    }

    std::cout << "[INFO] 正在加载 Embedding 模型: " << config.model.emb_model_path << std::endl;
    EmbeddingEngine embEngine(config.model.emb_model_path,
                              config.embedding, config.generation,
                              config.model.n_gpu_layers);

    std::cout << "[INFO] 正在生成向量嵌入..." << std::endl;
    VectorDatabase db;
    for (size_t i = 0; i < chunks.size(); ++i) {
        assert(!chunks[i].text.empty());
        chunks[i].embedding = embEngine.generateEmbedding(chunks[i].text);
        assert(!chunks[i].embedding.empty());
        if ((i + 1) % 10 == 0 || i + 1 == chunks.size()) {
            std::cout << "\r[INFO] 嵌入进度: " << (i + 1) << "/" << chunks.size() << std::flush;
        }
    }
    std::cout << std::endl;

    std::cout << "[INFO] 正在写入向量数据库..." << std::endl;
    db.insert(std::move(chunks));
    if (!db.saveToDisk(dbPath)) {
        throw std::runtime_error("无法保存数据库到: " + dbPath);
    }
    std::cout << "[INFO] 知识库构建完成，保存至: " << dbPath << std::endl;
    std::cout << "[INFO] 数据库包含 " << db.size() << " 个向量条目" << std::endl;
}

// 交互式问答循环：加载 DB → 初始化引擎 → 每轮前后自动检测热切换
static void chatLoop(const std::string &dbPath, const AppConfig &config) {
    assert(!dbPath.empty());
    std::cout << "[INFO] 正在加载向量数据库: " << dbPath << std::endl;
    VectorDatabase db;
    if (!db.loadFromDisk(dbPath)) {
        throw std::runtime_error("无法加载数据库: " + dbPath);
    }
    std::cout << "[INFO] 数据库加载成功，共 " << db.size() << " 个片段" << std::endl;

    AppConfig activeConfig = config;

    std::cout << "[INFO] 正在加载 Embedding 模型: " << activeConfig.model.emb_model_path << std::endl;
    auto embEngine = std::make_unique<EmbeddingEngine>(activeConfig.model.emb_model_path,
                                                       activeConfig.embedding, activeConfig.generation,
                                                       activeConfig.model.n_gpu_layers);

    std::cout << "[INFO] 正在加载生成模型: " << activeConfig.model.gen_model_path << std::endl;
    auto genEngine = std::make_unique<GenerationEngine>(activeConfig.model.gen_model_path,
                                                        activeConfig.generation, activeConfig.embedding,
                                                        activeConfig.model.n_gpu_layers);

    std::cout << "\n========== RAG 问答系统就绪 ==========" << std::endl;
    std::cout << "输入问题后按回车提交，输入 /reload 交互式切换模型，输入 /exit 退出" << std::endl;
    std::cout << "======================================\n" << std::endl;

    std::string query;
    while (true) {
        std::cout << "> " << std::flush;
        if (!std::getline(std::cin, query)) break;
        query = trimCommandInput(query);
        if (query.empty()) continue;
        if (query == "/exit") break;
        if (query == "/reload") {
            try {
                interactiveReload(activeConfig, db, embEngine, genEngine);
            } catch (const std::exception &e) {
                std::cerr << "[ERROR] 模型热切换失败: " << e.what() << std::endl;
                if (!embEngine) {
                    std::cerr << "[INFO] 正在恢复原 Embedding 模型..." << std::endl;
                    embEngine = std::make_unique<EmbeddingEngine>(activeConfig.model.emb_model_path,
                                                                  activeConfig.embedding, activeConfig.generation,
                                                                  activeConfig.model.n_gpu_layers);
                }
                if (!genEngine) {
                    std::cerr << "[INFO] 正在恢复原生成模型..." << std::endl;
                    genEngine = std::make_unique<GenerationEngine>(activeConfig.model.gen_model_path,
                                                                   activeConfig.generation, activeConfig.embedding,
                                                                   activeConfig.model.n_gpu_layers);
                }
                std::cerr << "[INFO] 程序不会退出；可以重新输入 /reload 或 /exit。" << std::endl;
            }
            continue;
        }
        if (!query.empty() && query[0] == '/') {
            std::cout << "[WARN] 未知指令: " << query << std::endl;
            std::cout << "[INFO] 可用指令: /reload, /exit" << std::endl;
            continue;
        }

        try {
            reloadFromConfig(activeConfig, db, embEngine, genEngine);
            doQuery(db, *embEngine, *genEngine, query, activeConfig);
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] 本轮问答失败: " << e.what() << std::endl;
            std::cerr << "[INFO] 可以继续提问，或输入 /reload 切换模型，输入 /exit 退出。" << std::endl;
        }
    }
}

// 单次查询模式：加载 DB → 生成 Embedding → 检索 → 流式输出 → 退出
static void singleQuery(const std::string &dbPath, const std::string &query, const AppConfig &config) {
    VectorDatabase db;
    if (!db.loadFromDisk(dbPath)) {
        throw std::runtime_error("无法加载数据库: " + dbPath);
    }

    EmbeddingEngine embEngine(config.model.emb_model_path,
                              config.embedding, config.generation,
                              config.model.n_gpu_layers);

    GenerationEngine genEngine(config.model.gen_model_path,
                               config.generation, config.embedding,
                               config.model.n_gpu_layers);

    doQuery(db, embEngine, genEngine, query, config);
}

static void doQuery(VectorDatabase &db, EmbeddingEngine &embEngine, GenerationEngine &genEngine,
                    const std::string &query, const AppConfig &config) {
    assert(!query.empty());
    assert(db.size() > 0);
    std::vector<float> queryEmb = embEngine.generateEmbedding(query);
    assert(!queryEmb.empty());

    std::vector<Chunk> results = db.search(queryEmb, query, config.retrieval.top_k);

    std::cout << "\n[数据库共 " << db.size() << " 个片段，Top-" << results.size() << " 最相关结果如下]" << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        auto preview = utf8Preview(results[i].text, 80);
        std::cout << "  [" << (i+1) << "] " << results[i].metadata << ": " << preview << "..." << std::endl;
    }

    if (results.empty()) {
        std::cout << "[WARN] 没有检索到可用片段，跳过生成。" << std::endl;
        return;
    }

    std::cout << "[INFO] 正在生成回答..." << std::endl;
    std::string answer = genEngine.generateStream(query, results);

    std::cout << answer << std::endl;
    std::cout << std::endl;
}

// 打印命令行用法，列出 --ingest / --chat / --query 三种模式
static void printUsage(const char *progName) {
    std::cerr << "用法:" << std::endl;
    std::cerr << "  离线建库: " << progName << " --ingest <小说TXT路径...> <输出数据库路径>" << std::endl;
    std::cerr << "  在线问答: " << progName << " --chat <数据库路径>" << std::endl;
    std::cerr << "  单次查询: " << progName << " --query <数据库路径> \"问题\"" << std::endl;
}

// 入口：BackendGuard 最外层，解析 --ingest / --chat / --query 三种模式
int runMragApp(int argc, char *argv[]) {
    std::ios::sync_with_stdio(false);
    if (argc > 0) {
        gExecutablePath = argv[0];
    }

#ifdef _WIN32
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
#endif

    llama_log_set(nullLogCallback, nullptr);

    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    // BackendGuard backendGuard;  // temporarily disabled for debugging
    BackendGuard backendGuard;

    try {
        AppConfig config = loadConfig("config.json");

        std::string mode = argv[1];

        if (mode == "--probe-gen") {
            if (argc < 3) {
                return 1;
            }
            GenerationEngine probe(argv[2], config.generation, config.embedding,
                                   config.model.n_gpu_layers);
            return 0;

        } else if (mode == "--probe-emb") {
            if (argc < 3) {
                return 1;
            }
            EmbeddingEngine probe(argv[2], config.embedding, config.generation,
                                  config.model.n_gpu_layers);
            std::vector<float> test = probe.generateEmbedding("模型探测");
            return test.empty() ? 1 : 0;

        } else if (mode == "--ingest") {
            if (argc < 4) {
                printUsage(argv[0]);
                return 1;
            }
            std::vector<std::string> txtPaths;
            for (int i = 2; i < argc - 1; ++i) {
                txtPaths.emplace_back(argv[i]);
            }
            std::string dbPath = argv[argc - 1];
            buildKnowledgeBase(txtPaths, dbPath, config);

        } else if (mode == "--chat") {
            if (argc < 3) {
                printUsage(argv[0]);
                return 1;
            }
            chatLoop(argv[2], config);

        } else if (mode == "--query") {
            if (argc < 4) {
                printUsage(argv[0]);
                return 1;
            }
            singleQuery(argv[2], argv[3], config);

        } else {
            std::cerr << "[ERROR] 未知模式: " << mode << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    } catch (const std::exception &e) {
        std::cerr << "[FATAL] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
