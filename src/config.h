#pragma once

#include <string>
#include <cstdint>
#include <cstddef>

struct AppConfig {
    struct ModelConfig {
        std::string emb_model_path;
        std::string gen_model_path;
        int n_gpu_layers = 99;
    } model;

    struct DocumentConfig {
        size_t chunk_size = 500;
        size_t overlap_size = 50;
    } document;

    struct RetrievalConfig {
        int top_k = 3;
    } retrieval;

    struct GenerationConfig {
        int n_ctx = 4096;
        int n_batch = 4096;
        int n_ubatch = 512;
        int max_output_tokens = 512;
        float temperature = 0.7f;
        int top_k = 40;
        float top_p = 0.9f;
    } generation;

    struct EmbeddingConfig {
        int n_ctx = 512;
        int n_batch = 512;
        int n_ubatch = 512;
    } embedding;
};

// 从 config.json 读取配置，字段缺失时保留 C++ 默认值，不报错
AppConfig loadConfig(const std::string &configPath);
bool saveConfig(const std::string &configPath, const AppConfig &config);
