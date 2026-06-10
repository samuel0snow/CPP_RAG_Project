#include "embeddingengine.h"

#include <iostream>
#include <cstring>
#include <algorithm>
#include <cassert>

EmbeddingEngine::EmbeddingEngine(const std::string &modelPath,
                                 const AppConfig::EmbeddingConfig &embCfg,
                                 const AppConfig::GenerationConfig &genCfg,
                                 int nGpuLayers)
    : LlamaModelBase(modelPath, ModelMode::Embedding, genCfg, embCfg, nGpuLayers) {}

// 清理残缺 UTF-8 序列：遍历每个字节 → 判断首字节 → 验证后续字节 → 保留合法序列
std::string EmbeddingEngine::sanitizeUtf8(const std::string &text) {
    std::string result;
    result.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        size_t len = 0;
        if ((c & 0x80u) == 0u) {
            len = 1;
        } else if ((c & 0xE0u) == 0xC0u) {
            len = 2;
        } else if ((c & 0xF0u) == 0xE0u) {
            len = 3;
        } else if ((c & 0xF8u) == 0xF0u) {
            len = 4;
        } else {
            ++i;
            continue;
        }
        if (i + len <= text.size()) {
            bool valid = true;
            for (size_t j = 1; j < len; ++j) {
                if ((static_cast<unsigned char>(text[i + j]) & 0xC0u) != 0x80u) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                result.append(text, i, len);
            }
        }
        i += len;
    }
    return result;
}

// 文本 → Embedding：sanitize → tokenize → 截断 → clear KV → 构造 batch → decode → 取 embeddings
std::vector<float> EmbeddingEngine::generateEmbedding(const std::string &text) {
    assert(model_ != nullptr);
    assert(ctx_ != nullptr);
    std::string cleanText = sanitizeUtf8(text);
    if (cleanText.empty()) {
        int nEmbd = llama_model_n_embd(model_);
        return std::vector<float>(static_cast<size_t>(nEmbd), 0.0f);
    }

    std::vector<llama_token> tokens = tokenize(cleanText, true);
    assert(!tokens.empty());

    int maxTokens = std::min(nCtx_, nBatch_);
    assert(maxTokens > 0);
    if (static_cast<int>(tokens.size()) > maxTokens) {
        tokens.resize(static_cast<size_t>(maxTokens));
    }

    llama_memory_clear(llama_get_memory(ctx_), true);

    int32_t nTokens = static_cast<int32_t>(tokens.size());
    assert(nTokens > 0);
    llama_batch batch = llama_batch_init(nTokens, 0, 1);

    for (int32_t i = 0; i < nTokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == nTokens - 1) ? 1 : 0;
    }
    batch.n_tokens = nTokens;

    int ret;
    if (llama_model_has_encoder(model_)) {
        ret = llama_encode(ctx_, batch);
    } else {
        ret = llama_decode(ctx_, batch);
    }
    llama_batch_free(batch);

    if (ret != 0) {
        std::cerr << "[ERROR] Embedding inference 返回错误码: " << ret
                  << (llama_model_has_encoder(model_) ? " (encode)" : " (decode)") << std::endl;
        int nEmbd = llama_model_n_embd(model_);
        return std::vector<float>(static_cast<size_t>(nEmbd), 0.0f);
    }

    int nEmbd = llama_model_n_embd(model_);
    assert(nEmbd > 0);
    const float *embdData = nullptr;

    embdData = llama_get_embeddings_seq(ctx_, 0);
    if (!embdData) {
        embdData = llama_get_embeddings(ctx_);
    }

    if (!embdData) {
        return std::vector<float>(static_cast<size_t>(nEmbd), 0.0f);
    }

    return std::vector<float>(embdData, embdData + nEmbd);
}
