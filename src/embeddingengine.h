#pragma once

#include "llamamodel.h"

#include <string>
#include <vector>

class EmbeddingEngine : public LlamaModelBase {
public:
    EmbeddingEngine(const std::string &modelPath,
                    const AppConfig::EmbeddingConfig &embCfg,
                    const AppConfig::GenerationConfig &genCfg,
                    int nGpuLayers);

    // 文本 → Embedding 向量：截断 → clear KV → batch decode → 取 embeddings
    std::vector<float> generateEmbedding(const std::string &text);

private:
    // 清理残缺 UTF-8 序列（被截断的多字节字符等）
    static std::string sanitizeUtf8(const std::string &text);
};
