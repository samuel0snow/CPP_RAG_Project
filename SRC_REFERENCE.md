# 源码参考：函数、类与继承关系总览

> 覆盖 `src/` 下所有 `.cc` 文件中定义的函数、类之间的继承关系，以及函数间的调用链路。

---

## 一、类/结构体继承关系

```
Chunk (struct, 无继承)
AppConfig (struct, 含 5 个子结构体, 无继承)
DocumentProcessor (class, 无继承)
VectorDatabase (class, 无继承)
LlamaModelBase (class, 基类)
  ├── EmbeddingEngine : public LlamaModelBase
  └── GenerationEngine : public LlamaModelBase
BackendGuard (struct, mragapp.cc 局部定义, 无继承)
```

### 继承说明

- **`EmbeddingEngine`** 和 **`GenerationEngine`** 都继承自 **`LlamaModelBase`**。
- `LlamaModelBase` 提供 GGUF 模型加载、推理上下文创建、tokenize 等公共能力；子类根据 `ModelMode`（`Embedding` / `Generation`）设置不同的 `llama_context_params`。
- 拷贝构造、赋值、移动全部 `= delete`，只允许通过 `std::unique_ptr` 管理生命周期。

---

## 二、各 .cc 文件函数详解

### 1. chunk.cc —— Chunk 结构体 + 二进制读写

| 函数 | 可见性 | 作用 |
|---|---|---|
| `writeUint64(FILE*, uint64_t)` | static（文件内） | 写入 8 字节无符号整数 |
| `readUint64(FILE*)` | static（文件内） | 读取 8 字节无符号整数 |
| `writeString(FILE*, const std::string&)` | static（文件内） | 先写长度(8B)再写内容，长度=0 时跳过内容 |
| `readString(FILE*)` | static（文件内） | 先读长度(8B)再读内容，还原 std::string |
| `writeFloatVec(FILE*, const std::vector<float>&)` | static（文件内） | 先写元素个数(8B)再写 float 数组 |
| `readFloatVec(FILE*)` | static（文件内） | 先读元素个数(8B)再读 float 数组，还原 vector |
| `Chunk::serialize(FILE*) const` | public 成员 | 序列化格式：id → text_len+text → meta_len+meta → emb_count+emb_floats |
| `Chunk::deserialize(FILE*)` | public static 成员 | 按 serialize 逆序还原一个 Chunk |

**调用关系**：`writeUint64/readUint64` 被 `writeString/readString/writeFloatVec/readFloatVec` 调用；这 6 个文件内函数被 `serialize/deserialize` 调用。

---

### 2. config.cc —— 配置结构体 + JSON 解析

| 函数 | 可见性 | 作用 |
|---|---|---|
| `skipWhitespace(const char*&)` | static（文件内） | 跳过空格、制表、换行、回车 |
| `parseString(const char*&)` | static（文件内） | 解析 JSON 字符串（支持 `\n \t \r \\ \"` 转义） |
| `parseNumber(const char*&)` | static（文件内） | 解析整数或浮点数（含科学计数法） |
| `parseValue(const char*&)` | static（文件内） | 根据首字符分发到 parseString / parseObject / parseNumber |
| `parseObject(const char*&)` | static（文件内） | 递归下降解析 JSON 对象 `{ key: value, ... }` |
| `parseJson(const std::string&)` | static（文件内） | 入口：将 JSON 字符串解析为 JsonValue 树 |
| `loadModelConfig(const JsonValue&, ModelConfig&)` | static（文件内） | 填充 model 子结构体 |
| `loadDocumentConfig(const JsonValue&, DocumentConfig&)` | static（文件内） | 填充 document 子结构体 |
| `loadRetrievalConfig(const JsonValue&, RetrievalConfig&)` | static（文件内） | 填充 retrieval 子结构体 |
| `loadGenerationConfig(const JsonValue&, GenerationConfig&)` | static（文件内） | 填充 generation 子结构体 |
| `loadEmbeddingConfig(const JsonValue&, EmbeddingConfig&)` | static（文件内） | 填充 embedding 子结构体 |
| `loadConfig(const std::string&)` | **public** | 读取 config.json → 解析 → 填充 AppConfig；文件不存在/解析失败保留默认值 |
| `saveConfig(const std::string&, const AppConfig&)` | **public** | 将 AppConfig 写回 config.json（格式化输出） |

**调用关系**：`loadConfig` 调 `parseJson` → `parseValue` → `parseObject` → `parseString/parseNumber`，再调 5 个 `loadXxxConfig` 填充子结构体。

---

### 3. document.cc —— 文档切分

| 函数 | 可见性 | 作用 |
|---|---|---|
| `sourceNameFromPath(const std::string&)` | static（文件内） | 从文件路径提取纯文件名（去扩展名），如 `三国演义.txt` → `三国演义` |
| `DocumentProcessor::DocumentProcessor(const DocumentConfig&)` | public 构造 | 保存 chunkSize_ 和 overlapSize_ |
| `DocumentProcessor::nextUtf8Boundary(const std::string&, size_t)` | public static | 从 pos 往后找下一个合法的 UTF-8 字符起始字节（跳过续字节 10xxxxxx） |
| `DocumentProcessor::isChinesePunctuation(const std::string&, size_t)` | public static | 判断 pos 处是否为中文标点（。！？，；、）——3 字节 UTF-8 匹配 |
| `DocumentProcessor::isChapterTitle(const std::string&)` | public static | 正则匹配 `第X章/回/卷` 或 `Chapter N` 格式 |
| `DocumentProcessor::findCutPoint(const std::string&, size_t) const` | private | 从 targetPos 往回找最佳断点：优先 `\n` → 中文标点 → fallback 到 nextUtf8Boundary |
| `DocumentProcessor::processNovel(const std::string&)` | public | **核心入口**：读 TXT → 逐行判断章节标题 → 按 chunk_size 切分 → 找断点 → overlap 衔接 → 返回 `vector<Chunk>` |

**调用关系**：
- `processNovel` 是总入口，内部调用 `sourceNameFromPath`、`isChapterTitle`、`findCutPoint`。
- `findCutPoint` 调 `isChinesePunctuation` 和 `nextUtf8Boundary`。
- 每切一个 Chunk，用 `nextChunkId_++` 自增 ID，`metadata` 设为 `"书名 / 章节名"`。

**输出**：`std::vector<Chunk>`（此时 `embedding` 字段为空，后续由 `EmbeddingEngine` 填充）。

---

### 4. vectordatabase.cc —— 向量存储 + 相似度检索 + 磁盘持久化

#### 4.1 文件内辅助函数

| 函数 | 作用 |
|---|---|
| `splitUtf8Chars(const std::string&)` | 将 UTF-8 字符串按字符（1~4 字节）拆分 |
| `queryAliases(const std::string&)` | 查询实体别名表（如 "刘备"→"玄德,刘玄德"），返回相关词列表 |
| `entityCooccurrenceScore(query, chunkText)` | 多实体共现评分：查询中的实体是否在片段中出现（含别名归一），全部命中 +0.7，部分命中按比例 |
| `evidenceCueScore(query, chunkText)` | 题型证据加权：武器题、数量题、过程题等，根据关键词证据加权 |
| `keywordScore(query, chunkText)` | 关键词评分 = 去重单字密度²×0.25 + n-gram 匹配×0.55 + 别名匹配×0.20 |
| `extractSubjectBeforeDe(const std::string&)` | 提取"的"之前的主题词（如"刘备的字是什么"→"刘备"） |
| `isShortFactQuestion(const std::string&)` | 判断是否短事实题（含"是什么/是谁/哪里/在哪/哪些/多少"） |
| `subjectEvidenceScore(subject, chunkText)` | 主语证据评分：直接出现 +1.0，别名出现 +1.0，"姓X名Y"格式匹配 |
| `attributeEvidenceScore(query, chunkText)` | 属性证据评分：查询含"字/名/姓/号/武器"时，片段中对应关键词出现则加权 |
| `factQuestionBoost(query, chunkText)` | 综合短事实题加权：主语未出现 -0.25，证据充足 1.2~2.2，否则按比例 |

#### 4.2 公开成员函数

| 函数 | 作用 |
|---|---|
| `VectorDatabase::insert(const Chunk&)` | 单条插入（拷贝） |
| `VectorDatabase::insert(std::vector<Chunk>&&)` | 批量插入（移动语义，避免拷贝） |
| `VectorDatabase::dotProduct(a, b)` | 向量点积 |
| `VectorDatabase::vectorNorm(v)` | 向量 L2 范数 |
| `VectorDatabase::cosineSimilarity(a, b)` | 余弦相似度 = dot / (normA × normB) |
| `VectorDatabase::search(queryEmb, queryText, topK) const` | **核心检索**：混合评分 = 0.35×cosSim + 0.65×keyword + factBoost + cueBoost + entityBoost；用 `priority_queue` 最小堆维护 top-K（O(n log k)）；返回时拼接同章节前后相邻 Chunk 文本 |
| `VectorDatabase::saveToDisk(path) const` | 持久化：Magic(4B, `0x4D524147`) + Version(4B) + Count(8B) + N×Chunk序列化 |
| `VectorDatabase::loadFromDisk(path)` | 加载：校验魔数+版本号，不匹配拒绝加载；逐条 `Chunk::deserialize()` 还原 |

**调用关系**：
- `search` 调 `cosineSimilarity` → `dotProduct` + `vectorNorm`；调所有辅助评分函数。
- `saveToDisk` 调 `Chunk::serialize`。
- `loadFromDisk` 调 `Chunk::deserialize`。

---

### 5. llamamodel.cc —— llama.cpp 模型基类（RAII 封装）

| 函数 | 可见性 | 作用 |
|---|---|---|
| `LlamaModelBase::LlamaModelBase(modelPath, mode, genCfg, embCfg, nGpuLayers)` | public 构造 | 加载 GGUF → 根据 `ModelMode` 设 `embeddings` 开关 → 创建 context → 拿 vocab。Embedding 模式用 `embCfg` 参数，Generation 模式用 `genCfg` 参数 |
| `LlamaModelBase::~LlamaModelBase()` | virtual public 析构 | 先 `llama_free(ctx_)` 再 `llama_model_free(model_)`，顺序不可颠倒 |
| `LlamaModelBase::tokenize(text, addBos) const` | protected | **两次 tokenize**：第一次 `nullptr` 拿数量 → 分配 vector → 第二次拿真实 token 列表，保证不溢出 |

**保护成员变量**：`model_`、`ctx_`、`vocab_`、`mode_`、`nCtx_`、`nBatch_`、`nUbatch_`。子类通过继承访问。

---

### 6. embeddingengine.cc —— 文本 → Embedding 向量

| 函数 | 可见性 | 作用 |
|---|---|---|
| `EmbeddingEngine::EmbeddingEngine(modelPath, embCfg, genCfg, nGpuLayers)` | public 构造 | 调用 `LlamaModelBase(path, Embedding, ...)` 初始化 |
| `EmbeddingEngine::sanitizeUtf8(text)` | private static | 清洗残缺 UTF-8 序列（遍历字节 → 判断首字节长度 → 验证续字节 → 保留合法序列） |
| `EmbeddingEngine::generateEmbedding(text)` | public | **核心**：sanitize → 调 `tokenize()` → 截断到 min(nCtx, nBatch) → `llama_memory_clear` 清 KV → 构造 batch（末 token 开 logits） → `llama_encode` 或 `llama_decode` → 优先 `llama_get_embeddings_seq`，fallback `llama_get_embeddings` → 返回 n_embd 维 float 向量 |

**调用关系**：`generateEmbedding` 调 `sanitizeUtf8` 和父类 `tokenize`。

---

### 7. generationengine.cc —— RAG 提示词构造 + 流式生成

| 函数 | 可见性 | 作用 |
|---|---|---|
| `buildContextString(chunks)` | static（文件内） | 将检索到的 Chunk 格式化为 `[片段N，出自：第X回]\n内容\n` |
| `buildChatMLPrompt(query, context)` | static（文件内） | 用 ChatML 模板组装 prompt：system（只根据上下文回答） + user（参考上下文 + 问题） + assistant |
| `GenerationEngine::GenerationEngine(...)` | public 构造 | 调父类构造 + 保存 maxOutputTokens/temperature/topKSampler/topP |
| `GenerationEngine::generateStream(query, chunks)` | public | **核心流式生成**：构建 prompt → tokenize → 超出上下文则拒绝 → clear KV → prefill batch → `llama_decode` → 初始化采样链（top_k → top_p → temperature → dist） → 自回归循环（采 token → 判 EOG → 解码输出 → 单 token batch → decode） → 最多 maxOutputTokens 个 |

**调用关系**：`generateStream` 调 `buildContextString`、`buildChatMLPrompt` 和父类 `tokenize`。

---

### 8. mragapp.cc —— 应用层

#### 8.1 结构体

| 结构体 | 作用 |
|---|---|
| `BackendGuard` | RAII 管理 `llama_backend_init/free` 生命周期，构造时 init，析构时 free。无继承。 |

#### 8.2 文件内辅助函数

| 函数 | 作用 |
|---|---|
| `nullLogCallback(level, text, user_data)` | 抑制 llama.cpp 的默认日志输出 |
| `trimCommandInput(input)` | 去除首尾空白 + 去除 BOM 头（`EF BB BF`） |
| `containsIgnoreCase(text, needle)` | 大小写不敏感的子串匹配 |
| `utf8Preview(text, maxChars)` | 截取前 maxChars 个 UTF-8 字符作为预览 |
| `modelPathFromName(modelsDir, fileName)` | 拼接模型目录路径 |
| `samePathText(lhs, rhs)` | 标准化路径比较（`\`→`/`，去掉 `./` 前缀） |
| `quoteArg(arg)` | 给命令行参数加双引号（供 `std::system` 调用） |
| `probeEmbeddingModel(modelPath)` | 通过 `--probe-emb` 子进程探测 Embedding 模型是否可用 |
| `sameRuntimeModels(lhs, rhs)` | 比较两个 AppConfig 的模型路径和推理参数是否完全相同 |
| `printUsage(progName)` | 打印三种模式的命令行用法 |

#### 8.3 核心流程函数

| 函数 | 作用 |
|---|---|
| `buildKnowledgeBase(txtPaths, dbPath, config)` | **离线建库全流程**：逐个文档调 `processNovel()` → 合并所有 Chunk → 逐一调 `generateEmbedding()` → `VectorDatabase::insert(move)` → `saveToDisk()` |
| `doQuery(db, embEngine, genEngine, query, config)` | **单次问答流程**：query Embedding → `db.search()` → 打印片段预览 → `genEngine.generateStream()` → 输出答案 |
| `rebuildEmbeddings(db, engine)` | 用新 Embedding 引擎重建数据库中全部 Chunk 的向量 |
| `interactiveReload(activeConfig, db, embEngine, genEngine)` | **交互式 /reload**：扫描 models/ 目录 → 显示生成/Embedding 模型菜单 → 用户选择 → 热切换引擎（Embedding 切换时重建向量）→ 写回 config.json |
| `reloadFromConfig(activeConfig, db, embEngine, genEngine)` | 检测 config.json 变化 → 自动热切换模型/参数 |
| `chatLoop(dbPath, config)` | **交互式问答循环**：加载 DB → 初始化引擎 → 每轮调 `doQuery`，支持 `/reload`、`/exit`，每轮前后检测热切换 |
| `singleQuery(dbPath, query, config)` | **单次查询模式**：加载 DB → `doQuery` → 退出 |

#### 8.4 公开入口

| 函数 | 可见性 | 作用 |
|---|---|---|
| `runMragApp(argc, argv)` | public（声明于 mragapp.h） | **主入口**：BackendGuard RAII → 解析 `--ingest`/`--chat`/`--query`/`--probe-gen`/`--probe-emb` 模式 → 调度对应流程 |

---

### 9. main.cc —— 最小入口

| 函数 | 作用 |
|---|---|
| `main(argc, argv)` | 仅调用 `runMragApp(argc, argv)`，无其他逻辑 |

---

## 三、函数间调用关系图

```
main()
 └── runMragApp()
      ├── BackendGuard (RAII: llama_backend_init/free)
      ├── loadConfig("config.json")
      │    └── parseJson → parseValue → parseObject → parseString/parseNumber
      │        └── loadModelConfig / loadDocumentConfig / loadRetrievalConfig
      │            / loadGenerationConfig / loadEmbeddingConfig
      │
      ├── [--ingest] buildKnowledgeBase()
      │    ├── DocumentProcessor::processNovel(file)
      │    │    ├── sourceNameFromPath()
      │    │    ├── isChapterTitle()
      │    │    └── findCutPoint()
      │    │         ├── isChinesePunctuation()
      │    │         └── nextUtf8Boundary()
      │    ├── EmbeddingEngine::generateEmbedding()
      │    │    ├── sanitizeUtf8()
      │    │    └── LlamaModelBase::tokenize()
      │    ├── VectorDatabase::insert(move)
      │    └── VectorDatabase::saveToDisk()
      │         └── Chunk::serialize()
      │              └── writeUint64 / writeString / writeFloatVec
      │
      ├── [--chat] chatLoop()
      │    ├── VectorDatabase::loadFromDisk()
      │    │    └── Chunk::deserialize()
      │    │         └── readUint64 / readString / readFloatVec
      │    ├── EmbeddingEngine / GenerationEngine 构造 → LlamaModelBase 构造
      │    ├── doQuery()
      │    │    ├── EmbeddingEngine::generateEmbedding()
      │    │    ├── VectorDatabase::search()
      │    │    │    ├── cosineSimilarity → dotProduct + vectorNorm
      │    │    │    ├── keywordScore → splitUtf8Chars + queryAliases
      │    │    │    ├── factQuestionBoost → extractSubjectBeforeDe
      │    │    │    │    + subjectEvidenceScore + attributeEvidenceScore
      │    │    │    ├── evidenceCueScore
      │    │    │    └── entityCooccurrenceScore
      │    │    └── GenerationEngine::generateStream()
      │    │         ├── buildContextString()
      │    │         ├── buildChatMLPrompt()
      │    │         └── LlamaModelBase::tokenize()
      │    ├── interactiveReload()
      │    │    ├── probeEmbeddingModel() → --probe-emb 子进程
      │    │    ├── EmbeddingEngine / GenerationEngine 构造
      │    │    ├── rebuildEmbeddings()
      │    │    │    └── EmbeddingEngine::generateEmbedding()
      │    │    └── saveConfig()
      │    └── reloadFromConfig()
      │
      └── [--query] singleQuery()
           └── doQuery()  (同上)
```

---

## 四、类与类之间的依赖关系

```
Chunk (基础数据结构)
  ↑ 被以下模块依赖
  ├── DocumentProcessor::processNovel() 产出 vector<Chunk>
  ├── VectorDatabase 内部存储 vector<Chunk>
  ├── EmbeddingEngine::generateEmbedding() 填充 Chunk::embedding
  ├── GenerationEngine::generateStream() 接收 vector<Chunk>
  └── mragapp 所有流程都在搬运/转换 Chunk

AppConfig (配置中枢)
  ├── DocumentProcessor 用 DocumentConfig
  ├── LlamaModelBase 用 GenerationConfig + EmbeddingConfig
  ├── VectorDatabase::search 用 RetrievalConfig::top_k
  └── mragapp 用全部子配置创建各引擎

LlamaModelBase (RAII 模型基类)
  ↑ 继承
  ├── EmbeddingEngine → 提供 generateEmbedding()
  └── GenerationEngine → 提供 generateStream()
  ↑ 被使用
  └── mragapp 通过 unique_ptr<EmbeddingEngine/GenerationEngine> 管理

VectorDatabase
  ↑ 被 mragapp 使用（建库写入、问答检索）
  └── 内部依赖 Chunk 的 serialize/deserialize 做持久化
```

---

## 五、数据流总览

```
TXT 文件
  │ DocumentProcessor::processNovel()
  ▼
vector<Chunk> (embedding 为空)
  │ EmbeddingEngine::generateEmbedding()
  ▼
vector<Chunk> (embedding 已填充)
  │ VectorDatabase::insert() + saveToDisk()
  ▼
.db 文件 (磁盘持久化)
  │ VectorDatabase::loadFromDisk()
  ▼
VectorDatabase (内存)
  │ 用户输入问题
  │   ├── EmbeddingEngine::generateEmbedding(query)
  │   └── VectorDatabase::search()
  ▼
vector<Chunk> (检索结果)
  │ GenerationEngine::generateStream()
  ▼
std::string (生成的回答)
```
