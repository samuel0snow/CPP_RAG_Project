#pragma once

#include "llamamodel.h"
#include "chunk.h"

#include <string>
#include <vector>
#include <ctime>

class GenerationEngine : public LlamaModelBase {
public:
    GenerationEngine(const std::string &modelPath,
                     const AppConfig::GenerationConfig &genCfg,
                     const AppConfig::EmbeddingConfig &embCfg,
                     int nGpuLayers);

    // 流式 RAG 生成：格式化上下文 → ChatML Prompt → prefill → 采样链循环生成
    std::string generateStream(const std::string &query,
                                const std::vector<Chunk> &chunks);

private:
    int maxOutputTokens_;
    float temperature_;
    int topKSampler_;
    float topP_;
};
