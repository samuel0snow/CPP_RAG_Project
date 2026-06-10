#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include "llama.h"
#include "config.h"

enum class ModelMode { Embedding, Generation };

class LlamaModelBase {
public:
    // 加载 GGUF 模型 → 创建推理上下文 → 获取 vocab，失败抛 runtime_error
    LlamaModelBase(const std::string &modelPath, ModelMode mode,
                   const AppConfig::GenerationConfig &genCfg,
                   const AppConfig::EmbeddingConfig &embCfg,
                   int nGpuLayers);
    // RAII：先 free ctx_，再 free model_，顺序不可颠倒
    virtual ~LlamaModelBase();

    LlamaModelBase(const LlamaModelBase &) = delete;
    LlamaModelBase &operator=(const LlamaModelBase &) = delete;
    LlamaModelBase(LlamaModelBase &&) = delete;
    LlamaModelBase &operator=(LlamaModelBase &&) = delete;

protected:
    llama_model *model_ = nullptr;
    llama_context *ctx_ = nullptr;
    const llama_vocab *vocab_ = nullptr;
    ModelMode mode_;
    int nCtx_ = 0;
    int nBatch_ = 0;
    int nUbatch_ = 0;

    // 两次 llama_tokenize：先取 token 数量，再取 token 列表，保证不溢出
    std::vector<llama_token> tokenize(const std::string &text, bool addBos = true) const;
};
