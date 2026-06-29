#include "config.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

enum class JsonType { String, Integer, Float, Object };

struct JsonValue {
    JsonType type;
    std::string strVal;
    int intVal = 0;
    float floatVal = 0.0f;
    std::unordered_map<std::string, JsonValue> objVal;

    bool has(const std::string &key) const {
        return type == JsonType::Object && objVal.find(key) != objVal.end();
    }

    const JsonValue &operator[](const std::string &key) const {
        static JsonValue empty;
        auto it = objVal.find(key);
        return it != objVal.end() ? it->second : empty;
    }

    std::string getString() const { return strVal; }
    int getInt() const { return intVal; }
    float getFloat() const { return floatVal; }
};

static void skipWhitespace(const char *&p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
}

static std::string parseString(const char *&p) {
    if (*p != '"') return "";
    ++p;
    std::string result;
    while (*p && *p != '"') {
        if (*p == '\\') {
            ++p;
            if (*p == '"') result += '"';
            else if (*p == '\\') result += '\\';
            else if (*p == '/') result += '/';
            else if (*p == 'n') result += '\n';
            else if (*p == 't') result += '\t';
            else if (*p == 'r') result += '\r';
            else { result += '\\'; result += *p; }
        } else {
            result += *p;
        }
        ++p;
    }
    if (*p == '"') ++p;
    return result;
}

static JsonValue parseNumber(const char *&p) {
    JsonValue v;
    bool isFloat = false;
    std::string numStr;
    if (*p == '-') { numStr += '-'; ++p; }
    while (*p >= '0' && *p <= '9') { numStr += *p; ++p; }
    if (*p == '.') {
        isFloat = true;
        numStr += '.';
        ++p;
        while (*p >= '0' && *p <= '9') { numStr += *p; ++p; }
    }
    if (*p == 'e' || *p == 'E') {
        isFloat = true;
        numStr += *p; ++p;
        if (*p == '+' || *p == '-') { numStr += *p; ++p; }
        while (*p >= '0' && *p <= '9') { numStr += *p; ++p; }
    }
    if (isFloat) {
        v.type = JsonType::Float;
        v.floatVal = static_cast<float>(std::atof(numStr.c_str()));
    } else {
        v.type = JsonType::Integer;
        v.intVal = std::atoi(numStr.c_str());
    }
    return v;
}

static JsonValue parseValue(const char *&p);

static JsonValue parseObject(const char *&p) {
    JsonValue v;
    v.type = JsonType::Object;
    if (*p != '{') return v;
    ++p;
    skipWhitespace(p);
    if (*p == '}') { ++p; return v; }
    while (true) {
        skipWhitespace(p);
        std::string key = parseString(p);
        skipWhitespace(p);
        if (*p == ':') ++p;
        skipWhitespace(p);
        v.objVal[key] = parseValue(p);
        skipWhitespace(p);
        if (*p == '}') { ++p; break; }
        if (*p == ',') ++p;
    }
    return v;
}

static JsonValue parseValue(const char *&p) {
    skipWhitespace(p);
    if (*p == '"') {
        JsonValue v;
        v.type = JsonType::String;
        v.strVal = parseString(p);
        return v;
    }
    if (*p == '{') {
        return parseObject(p);
    }
    if ((*p >= '0' && *p <= '9') || *p == '-') {
        return parseNumber(p);
    }
    return JsonValue{};
}

static JsonValue parseJson(const std::string &text) {
    const char *p = text.c_str();
    return parseValue(p);
}

static void loadModelConfig(const JsonValue &j, AppConfig::ModelConfig &cfg) {
    if (j.has("model")) {
        const auto &m = j["model"];
        if (m.has("embedding")) cfg.emb_model_path = m["embedding"].getString();
        if (m.has("generation")) cfg.gen_model_path = m["generation"].getString();
        if (m.has("n_gpu_layers")) cfg.n_gpu_layers = m["n_gpu_layers"].getInt();
    }
}

static void loadDocumentConfig(const JsonValue &j, AppConfig::DocumentConfig &cfg) {
    if (j.has("document")) {
        const auto &d = j["document"];
        if (d.has("chunk_size")) cfg.chunk_size = static_cast<size_t>(d["chunk_size"].getInt());
        if (d.has("overlap_size")) cfg.overlap_size = static_cast<size_t>(d["overlap_size"].getInt());
    }
}

static void loadRetrievalConfig(const JsonValue &j, AppConfig::RetrievalConfig &cfg) {
    if (j.has("retrieval")) {
        const auto &r = j["retrieval"];
        if (r.has("top_k")) cfg.top_k = r["top_k"].getInt();
    }
}

static void loadGenerationConfig(const JsonValue &j, AppConfig::GenerationConfig &cfg) {
    if (j.has("generation")) {
        const auto &g = j["generation"];
        if (g.has("n_ctx")) cfg.n_ctx = g["n_ctx"].getInt();
        if (g.has("n_batch")) cfg.n_batch = g["n_batch"].getInt();
        if (g.has("n_ubatch")) cfg.n_ubatch = g["n_ubatch"].getInt();
        if (g.has("max_output_tokens")) cfg.max_output_tokens = g["max_output_tokens"].getInt();
        if (g.has("temperature")) cfg.temperature = g["temperature"].getFloat();
        if (g.has("top_k")) cfg.top_k = g["top_k"].getInt();
        if (g.has("top_p")) cfg.top_p = g["top_p"].getFloat();
    }
}

static void loadEmbeddingConfig(const JsonValue &j, AppConfig::EmbeddingConfig &cfg) {
    if (j.has("embedding")) {
        const auto &e = j["embedding"];
        if (e.has("n_ctx")) cfg.n_ctx = e["n_ctx"].getInt();
        if (e.has("n_batch")) cfg.n_batch = e["n_batch"].getInt();
        if (e.has("n_ubatch")) cfg.n_ubatch = e["n_ubatch"].getInt();
    }
}

// 读取 JSON → 手写递归下降解析 → 填充五个子结构体，文件不存在或解析失败保留默认值
AppConfig loadConfig(const std::string &configPath) {
    AppConfig config;

    FILE *fp = fopen(configPath.c_str(), "r");
    if (!fp) {
        return config;
    }

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    std::string buffer(static_cast<size_t>(fsize), '\0');
    fread(&buffer[0], 1, static_cast<size_t>(fsize), fp);
    fclose(fp);

    try {
        JsonValue j = parseJson(buffer);
        loadModelConfig(j, config.model);
        loadDocumentConfig(j, config.document);
        loadRetrievalConfig(j, config.retrieval);
        loadGenerationConfig(j, config.generation);
        loadEmbeddingConfig(j, config.embedding);
    } catch (...) {
        std::cerr << "[WARN] JSON 解析失败，使用默认配置" << std::endl;
    }
    return config;
}

// 将当前运行配置写回 config.json，供热切换后的下轮检测保持一致
bool saveConfig(const std::string &configPath, const AppConfig &config) {
    FILE *fp = fopen(configPath.c_str(), "w");
    if (!fp) {
        std::cerr << "[ERROR] 无法写入配置文件: " << configPath << std::endl;
        return false;
    }

    int written = std::fprintf(
        fp,
        "{\n"
        "    \"model\": {\n"
        "        \"embedding\": \"%s\",\n"
        "        \"generation\": \"%s\",\n"
        "        \"n_gpu_layers\": %d\n"
        "    },\n"
        "    \"document\": {\n"
        "        \"chunk_size\": %zu,\n"
        "        \"overlap_size\": %zu\n"
        "    },\n"
        "    \"retrieval\": {\n"
        "        \"top_k\": %d\n"
        "    },\n"
        "    \"generation\": {\n"
        "        \"n_ctx\": %d,\n"
        "        \"n_batch\": %d,\n"
        "        \"n_ubatch\": %d,\n"
        "        \"max_output_tokens\": %d,\n"
        "        \"temperature\": %.6g,\n"
        "        \"top_k\": %d,\n"
        "        \"top_p\": %.6g\n"
        "    },\n"
        "    \"embedding\": {\n"
        "        \"n_ctx\": %d,\n"
        "        \"n_batch\": %d,\n"
        "        \"n_ubatch\": %d\n"
        "    }\n"
        "}\n",
        config.model.emb_model_path.c_str(),
        config.model.gen_model_path.c_str(),
        config.model.n_gpu_layers,
        config.document.chunk_size,
        config.document.overlap_size,
        config.retrieval.top_k,
        config.generation.n_ctx,
        config.generation.n_batch,
        config.generation.n_ubatch,
        config.generation.max_output_tokens,
        config.generation.temperature,
        config.generation.top_k,
        config.generation.top_p,
        config.embedding.n_ctx,
        config.embedding.n_batch,
        config.embedding.n_ubatch);

    bool ok = written > 0 && ferror(fp) == 0;
    fclose(fp);
    return ok;
}
