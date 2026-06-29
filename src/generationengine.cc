#include "generationengine.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <cassert>

GenerationEngine::GenerationEngine(const std::string &modelPath,
                                   const AppConfig::GenerationConfig &genCfg,
                                   const AppConfig::EmbeddingConfig &embCfg,
                                   int nGpuLayers)
    : LlamaModelBase(modelPath, ModelMode::Generation, genCfg, embCfg, nGpuLayers),
      maxOutputTokens_(genCfg.max_output_tokens),
      temperature_(genCfg.temperature),
      topKSampler_(genCfg.top_k),
      topP_(genCfg.top_p) {}

// 将检索到的 Chunk 列表格式化为带出处标注的上下文字符串
static std::string buildContextString(const std::vector<Chunk> &chunks) {
    assert(!chunks.empty());
    std::ostringstream oss;
    for (size_t i = 0; i < chunks.size(); ++i) {
        oss << "[片段" << (i + 1) << "，出自：" << chunks[i].metadata << "]\n";
        oss << chunks[i].text << "\n\n";
    }
    return oss.str();
}

// 用 ChatML 模板组装完整 Prompt：system(只根据上下文回答) + user(上下文+问题) + assistant
static std::string buildChatMLPrompt(const std::string &query, const std::string &context) {
    assert(!query.empty());
    assert(!context.empty());
    std::ostringstream oss;
    oss << "<|im_start|>system\n";
    oss << "你是一个严谨的小说阅读助手。请你【必须并且只能】根据下面提供的[参考上下文]来回答用户的问题。\n";
    oss << "回答前先检查上下文是否出现能直接支持答案的原文证据；";
    oss << "如果证据不足、人物不匹配或只看到相似名字，必须回答\"根据当前片段无法得到答案\"，不要猜测。\n";
    oss << "如果上下文中有答案，请简明回答并引用出处；";
    oss << "短事实问题（如字、名、姓、武器、地点、人物）只回答被原文明确支持的内容；";
    oss << "回答时保留用户问题中的主体，例如问\"刘备的字是什么\"，应回答\"刘备的字是玄德\"。";
    oss << "<|im_end|>\n";
    oss << "<|im_start|>user\n";
    oss << "[参考上下文]：\n";
    oss << context << "\n";
    oss << "用户问题：" << query << "<|im_end|>\n";
    oss << "<|im_start|>assistant\n";
    return oss.str();
}

// 流式 RAG 生成：构建 Prompt → tokenize → clear KV → prefill → 采样链循环 → 检测 EOG 停止
std::string GenerationEngine::generateStream(const std::string &query,
                                              const std::vector<Chunk> &chunks) {
    assert(model_ != nullptr);
    assert(ctx_ != nullptr);
    assert(!query.empty());
    std::string contextStr = buildContextString(chunks);
    std::string prompt = buildChatMLPrompt(query, contextStr);

    std::vector<llama_token> tokens = tokenize(prompt, true);
    assert(!tokens.empty());

    int maxCtx = nCtx_;
    assert(maxCtx > 0);
    if (static_cast<int>(tokens.size()) > maxCtx) {
        std::cerr << "[WARN] Prompt 超出上下文窗口 (" << tokens.size()
                  << " > " << maxCtx << ")，跳过本次推理" << std::endl;
        return "[错误] 输入上下文过长，请缩短问题或减少检索片段数。";
    }

    llama_memory_clear(llama_get_memory(ctx_), true);

    int32_t nTokens = static_cast<int32_t>(tokens.size());
    assert(nTokens > 0);
    llama_batch prefillBatch = llama_batch_init(nTokens, 0, 1);

    for (int32_t i = 0; i < nTokens; ++i) {
        prefillBatch.token[i] = tokens[i];
        prefillBatch.pos[i] = i;
        prefillBatch.n_seq_id[i] = 1;
        prefillBatch.seq_id[i][0] = 0;
        prefillBatch.logits[i] = (i == nTokens - 1) ? 1 : 0;
    }
    prefillBatch.n_tokens = nTokens;

    int ret = llama_decode(ctx_, prefillBatch);
    llama_batch_free(prefillBatch);

    if (ret != 0) {
        std::cerr << "[ERROR] llama_decode prefill 返回错误码: " << ret << std::endl;
        return "[错误] 模型推理失败。";
    }

    llama_sampler_chain_params samplerParams = llama_sampler_chain_default_params();
    llama_sampler *sampler = llama_sampler_chain_init(samplerParams);
    assert(sampler != nullptr);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(
        static_cast<int32_t>(topKSampler_)));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(topP_, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature_));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(
        static_cast<uint32_t>(std::time(nullptr))));

    std::string generatedText;

    for (int i = 0; i < maxOutputTokens_; ++i) {
        llama_token newToken = llama_sampler_sample(sampler, ctx_, -1);

        if (llama_vocab_is_eog(vocab_, newToken)) {
            break;
        }

        char buf[256];
        int nChars = llama_token_to_piece(vocab_, newToken, buf, sizeof(buf), 0, true);
        if (nChars > 0) {
            generatedText.append(buf, static_cast<size_t>(nChars));
        }

        llama_batch singleBatch = llama_batch_init(1, 0, 1);
        singleBatch.token[0] = newToken;
        singleBatch.pos[0] = nTokens + i;
        singleBatch.n_seq_id[0] = 1;
        singleBatch.seq_id[0][0] = 0;
        singleBatch.logits[0] = 1;
        singleBatch.n_tokens = 1;
        ret = llama_decode(ctx_, singleBatch);
        llama_batch_free(singleBatch);
        if (ret != 0) {
            std::cerr << "[ERROR] llama_decode 自回归返回错误码: " << ret << std::endl;
            break;
        }
    }

    llama_sampler_free(sampler);
    if (generatedText.empty()) {
        return "[提示] 模型没有生成有效内容，请换个问法或输入 /reload 切换模型后再试。";
    }
    return generatedText;
}
