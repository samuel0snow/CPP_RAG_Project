#include "llamamodel.h"

#include <cassert>
#include <stdexcept>
#include <iostream>

// 加载 GGUF 模型 → 根据 ModelMode 设 embeddings 开关 → 创建 context → 拿 vocab
LlamaModelBase::LlamaModelBase(const std::string &modelPath, ModelMode mode,
                               const AppConfig::GenerationConfig &genCfg,
                               const AppConfig::EmbeddingConfig &embCfg,
                               int nGpuLayers)
    : mode_(mode) {
    llama_model_params modelParams = llama_model_default_params();
    assert(!modelPath.empty());
    assert(nGpuLayers >= 0);
    modelParams.n_gpu_layers = nGpuLayers;

    model_ = llama_model_load_from_file(modelPath.c_str(), modelParams);
    if (!model_) {
        throw std::runtime_error("无法加载模型文件: " + modelPath);
    }

    llama_context_params ctxParams = llama_context_default_params();

    if (mode == ModelMode::Embedding) {
        ctxParams.embeddings = true;
        nCtx_ = embCfg.n_ctx;
        nBatch_ = embCfg.n_batch;
        nUbatch_ = embCfg.n_ubatch;
    } else {
        ctxParams.embeddings = false;
        nCtx_ = genCfg.n_ctx;
        nBatch_ = genCfg.n_batch;
        nUbatch_ = genCfg.n_ubatch;
    }
    assert(nCtx_ > 0);
    assert(nBatch_ > 0);
    assert(nUbatch_ > 0);
    assert(nUbatch_ <= nBatch_);

    ctxParams.n_ctx = static_cast<uint32_t>(nCtx_);
    ctxParams.n_batch = static_cast<uint32_t>(nBatch_);
    ctxParams.n_ubatch = static_cast<uint32_t>(nUbatch_);

    ctx_ = llama_init_from_model(model_, ctxParams);
    if (!ctx_) {
        llama_model_free(model_);
        model_ = nullptr;
        throw std::runtime_error("无法创建推理上下文: " + modelPath);
    }

    vocab_ = llama_model_get_vocab(model_);
    assert(vocab_ != nullptr);
}

LlamaModelBase::~LlamaModelBase() {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}

// 两次 tokenize：① nullptr 取数量 → ② 分配 vector 后取真实 token 列表
std::vector<llama_token> LlamaModelBase::tokenize(const std::string &text, bool addBos) const {
    assert(vocab_ != nullptr);
    assert(!text.empty());
    int nTokens = -llama_tokenize(vocab_, text.c_str(), static_cast<int>(text.size()),
                                   nullptr, 0, addBos, true);
    if (nTokens < 0) {
        throw std::runtime_error("tokenize 失败: 无法获取 token 数量");
    }

    std::vector<llama_token> tokens(static_cast<size_t>(nTokens));
    int actual = llama_tokenize(vocab_, text.c_str(), static_cast<int>(text.size()),
                                 tokens.data(), static_cast<int>(tokens.size()), addBos, true);
    if (actual < 0) {
        throw std::runtime_error("tokenize 失败: 无法获取 token 列表");
    }
    tokens.resize(static_cast<size_t>(actual));
    return tokens;
}
