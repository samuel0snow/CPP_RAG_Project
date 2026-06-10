#pragma once

#include "chunk.h"

#include <string>
#include <vector>
#include <queue>
#include <cmath>
#include <cstdint>

class VectorDatabase {
public:
    static constexpr uint32_t kMagic = 0x4D524147;
    static constexpr uint32_t kVersion = 1;

    // 插入单个/批量 Chunk，批量版本通过移动语义避免拷贝
    void insert(const Chunk &chunk);
    void insert(std::vector<Chunk> &&chunks);

    // 混合相似度检索：余弦相似度 × 0.5 + 关键词匹配 × 0.5，用最小堆 O(n log k)
    std::vector<Chunk> search(const std::vector<float> &queryEmb, const std::string &queryText, int topK) const;

    // 持久化：Magic(4B) + Version(4B) + Count(8B) + Chunk 序列化列表
    bool saveToDisk(const std::string &path) const;
    // 加载时校验魔数和版本号，不匹配则拒绝加载
    bool loadFromDisk(const std::string &path);

    size_t size() const { return chunks_.size(); }
    const std::vector<Chunk> &chunks() const { return chunks_; }

private:
    std::vector<Chunk> chunks_;

    static float dotProduct(const std::vector<float> &a, const std::vector<float> &b);
    static float vectorNorm(const std::vector<float> &v);
    // 余弦相似度：cos(A,B) = (A·B) / (|A| * |B|)
    static float cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b);
};
