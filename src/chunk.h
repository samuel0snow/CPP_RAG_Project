#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct Chunk {
    uint64_t id = 0;
    std::string text;
    std::string metadata;
    std::vector<float> embedding;

    Chunk() = default;
    Chunk(uint64_t id, std::string text, std::string metadata)
        : id(id), text(std::move(text)), metadata(std::move(metadata)) {}

    Chunk(Chunk &&) = default;
    Chunk &operator=(Chunk &&) = default;

    Chunk(const Chunk &) = default;
    Chunk &operator=(const Chunk &) = default;

    bool empty() const { return text.empty() && embedding.empty(); }

    // 将 Chunk 序列化为二进制：id → text → metadata → embedding
    void serialize(FILE *out) const;

    // 从二进制流还原 Chunk，顺序与 serialize 对应
    static Chunk deserialize(FILE *in);
};
