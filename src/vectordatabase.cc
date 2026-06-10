#include "vectordatabase.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <algorithm>

void VectorDatabase::insert(const Chunk &chunk) {
    assert(!chunk.text.empty());
    assert(!chunk.embedding.empty());
    chunks_.push_back(chunk);
}

void VectorDatabase::insert(std::vector<Chunk> &&chunks) {
    for (const auto &chunk : chunks) {
        assert(!chunk.text.empty());
        assert(!chunk.embedding.empty());
        (void)chunk;
    }
    chunks_.insert(chunks_.end(),
                   std::make_move_iterator(chunks.begin()),
                   std::make_move_iterator(chunks.end()));
}

float VectorDatabase::dotProduct(const std::vector<float> &a, const std::vector<float> &b) {
    assert(!a.empty());
    assert(a.size() == b.size());
    float result = 0.0f;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

float VectorDatabase::vectorNorm(const std::vector<float> &v) {
    assert(!v.empty());
    float sum = 0.0f;
    for (float x : v) sum += x * x;
    return std::sqrt(sum);
}

float VectorDatabase::cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b) {
    assert(!a.empty());
    assert(a.size() == b.size());
    float dot = dotProduct(a, b);
    float normA = vectorNorm(a);
    float normB = vectorNorm(b);
    if (normA == 0.0f || normB == 0.0f) return 0.0f;
    return dot / (normA * normB);
}

static float keywordScore(const std::string &query, const std::string &chunkText) {
    std::vector<std::string> chars;
    for (size_t i = 0; i < query.size(); ) {
        int clen = 1;
        unsigned char c = static_cast<unsigned char>(query[i]);
        if (c >= 0xE0) clen = 3;
        else if (c >= 0xC0) clen = 2;
        chars.push_back(query.substr(i, static_cast<size_t>(clen)));
        i += static_cast<size_t>(clen);
    }

    if (chars.empty()) return 0.0f;

    static const std::string stopChars = "的是在了也嗎呢啊吧與和及之乎者"
                                         "也者乎耶哉焉耳已矣夫";
    std::vector<std::string> distChars;
    for (const auto &ch : chars) {
        if (ch.size() == 3 && stopChars.find(ch) == std::string::npos) {
            distChars.push_back(ch);
        }
    }
    if (distChars.empty()) distChars = chars;

    int distTotal = 0, distFound = 0;
    for (const auto &ch : distChars) {
        distTotal++;
        if (chunkText.find(ch) != std::string::npos) distFound++;
    }
    float density = distTotal > 0 ? static_cast<float>(distFound) / static_cast<float>(distTotal) : 0.0f;

    int bigramTotal = 0, bigramFound = 0;
    for (size_t j = 0; j + 1 < chars.size(); ++j) {
        std::string bg = chars[j] + chars[j+1];
        bigramTotal++;
        if (chunkText.find(bg) != std::string::npos) bigramFound++;
    }
    float bigramScore = bigramTotal > 0 ? static_cast<float>(bigramFound) / static_cast<float>(bigramTotal) : 0.0f;

    return density * density * 0.6f + bigramScore * 0.4f;
}

static std::vector<std::string> splitUtf8Chars(const std::string &text) {
    std::vector<std::string> chars;
    for (size_t i = 0; i < text.size(); ) {
        size_t clen = 1;
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c >= 0xF0) clen = 4;
        else if (c >= 0xE0) clen = 3;
        else if (c >= 0xC0) clen = 2;
        if (i + clen > text.size()) break;
        chars.push_back(text.substr(i, clen));
        i += clen;
    }
    return chars;
}

static std::string extractSubjectBeforeDe(const std::string &query) {
    size_t de = query.find("的");
    if (de == std::string::npos || de == 0) return "";

    std::string prefix = query.substr(0, de);
    auto chars = splitUtf8Chars(prefix);
    while (!chars.empty()) {
        const std::string &ch = chars.front();
        if (ch == "请" || ch == "问" || ch == "说" || ch == "讲") {
            chars.erase(chars.begin());
        } else {
            break;
        }
    }
    while (!chars.empty() && chars.front() == "，") chars.erase(chars.begin());
    if (chars.empty() || chars.size() > 4) return "";

    std::string subject;
    for (const auto &ch : chars) subject += ch;
    return subject;
}

static bool isShortFactQuestion(const std::string &query) {
    return query.find("是什么") != std::string::npos ||
           query.find("是谁") != std::string::npos ||
           query.find("哪里") != std::string::npos ||
           query.find("在哪") != std::string::npos ||
           query.find("哪些") != std::string::npos ||
           query.find("多少") != std::string::npos;
}

static float subjectEvidenceScore(const std::string &subject, const std::string &chunkText) {
    if (subject.empty()) return 0.0f;
    if (chunkText.find(subject) != std::string::npos) return 1.0f;

    auto chars = splitUtf8Chars(subject);
    if (chars.size() == 2) {
        const std::string surname = chars[0];
        const std::string given = chars[1];
        if (chunkText.find("姓" + surname) != std::string::npos &&
            chunkText.find("名" + given) != std::string::npos) {
            return 1.0f;
        }
        if (chunkText.find("姓" + surname + "名" + given) != std::string::npos ||
            chunkText.find("姓" + surname + "，名" + given) != std::string::npos) {
            return 1.0f;
        }
        if (chunkText.find("名" + given) != std::string::npos) {
            return 0.6f;
        }
    }
    return 0.0f;
}

static float attributeEvidenceScore(const std::string &query, const std::string &chunkText) {
    float score = 0.0f;
    if (query.find("字") != std::string::npos && chunkText.find("字") != std::string::npos) score += 1.0f;
    if (query.find("名") != std::string::npos && chunkText.find("名") != std::string::npos) score += 0.8f;
    if (query.find("姓") != std::string::npos && chunkText.find("姓") != std::string::npos) score += 0.8f;
    if (query.find("号") != std::string::npos && chunkText.find("号") != std::string::npos) score += 0.8f;
    if ((query.find("武器") != std::string::npos || query.find("兵器") != std::string::npos) &&
        (chunkText.find("刀") != std::string::npos || chunkText.find("矛") != std::string::npos ||
         chunkText.find("剑") != std::string::npos || chunkText.find("斧") != std::string::npos)) {
        score += 0.8f;
    }
    return std::min(score, 1.5f);
}

static float factQuestionBoost(const std::string &query, const std::string &chunkText) {
    if (!isShortFactQuestion(query)) return 0.0f;
    std::string subject = extractSubjectBeforeDe(query);
    float subjectScore = subjectEvidenceScore(subject, chunkText);
    float attrScore = attributeEvidenceScore(query, chunkText);

    if (!subject.empty() && subjectScore == 0.0f) {
        return -0.25f;
    }
    if (subjectScore > 0.0f && attrScore > 0.0f) {
        return 1.2f + 0.6f * subjectScore + 0.4f * attrScore;
    }
    return 0.4f * subjectScore + 0.25f * attrScore;
}

// 混合相似度检索：cosSim × 0.5 + keyword × 0.5，用最小堆维护 top-K（O(n log k)）
std::vector<Chunk> VectorDatabase::search(const std::vector<float> &queryEmb, const std::string &queryText, int topK) const {
    assert(!queryEmb.empty());
    if (chunks_.empty() || topK <= 0) return {};

    using HeapEntry = std::pair<float, size_t>;
    auto cmp = [](const HeapEntry &a, const HeapEntry &b) { return a.first > b.first; };
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, decltype(cmp)> minHeap(cmp);

    for (size_t i = 0; i < chunks_.size(); ++i) {
        if (chunks_[i].embedding.size() != queryEmb.size()) {
            std::cerr << "[WARN] 跳过向量维度不一致的片段 id=" << chunks_[i].id
                      << " (" << chunks_[i].embedding.size()
                      << " != " << queryEmb.size() << ")" << std::endl;
            continue;
        }
        float cosSim = cosineSimilarity(queryEmb, chunks_[i].embedding);
        float kw = keywordScore(queryText, chunks_[i].text);
        float factBoost = factQuestionBoost(queryText, chunks_[i].text);
        float score = 0.45f * cosSim + 0.55f * kw + factBoost;
        if (static_cast<int>(minHeap.size()) < topK) {
            minHeap.emplace(score, i);
        } else if (score > minHeap.top().first) {
            minHeap.pop();
            minHeap.emplace(score, i);
        }
    }

    std::vector<Chunk> results;
    results.reserve(minHeap.size());
    std::vector<size_t> indices;
    indices.reserve(minHeap.size());
    while (!minHeap.empty()) {
        indices.push_back(minHeap.top().second);
        minHeap.pop();
    }
    for (auto it = indices.rbegin(); it != indices.rend(); ++it) {
        results.push_back(chunks_[*it]);
    }
    return results;
}

// 文件格式：Magic(4B) + Version(4B) + Count(8B) + N × Chunk 序列化
bool VectorDatabase::saveToDisk(const std::string &path) const {
    assert(!path.empty());
    FILE *fp = fopen(path.c_str(), "wb");
    if (!fp) {
        std::cerr << "[ERROR] 无法打开文件写入: " << path << std::endl;
        return false;
    }

    fwrite(&kMagic, sizeof(kMagic), 1, fp);
    fwrite(&kVersion, sizeof(kVersion), 1, fp);

    uint64_t count = static_cast<uint64_t>(chunks_.size());
    fwrite(&count, sizeof(count), 1, fp);

    for (const auto &chunk : chunks_) {
        assert(!chunk.text.empty());
        assert(!chunk.embedding.empty());
        chunk.serialize(fp);
    }

    bool ok = (ferror(fp) == 0);
    fclose(fp);
    return ok;
}

// 校验魔数 + 版本号，不匹配则拒绝加载并输出可读错误信息
bool VectorDatabase::loadFromDisk(const std::string &path) {
    assert(!path.empty());
    FILE *fp = fopen(path.c_str(), "rb");
    if (!fp) {
        std::cerr << "[ERROR] 无法打开文件读取: " << path << std::endl;
        return false;
    }

    uint32_t magic = 0, version = 0;
    fread(&magic, sizeof(magic), 1, fp);
    fread(&version, sizeof(version), 1, fp);

    if (magic != kMagic) {
        std::cerr << "[ERROR] 数据库文件魔数不匹配: 期望 0x"
                  << std::hex << kMagic << std::dec
                  << ", 实际 0x" << std::hex << magic << std::dec << std::endl;
        fclose(fp);
        return false;
    }
    if (version != kVersion) {
        std::cerr << "[ERROR] 数据库文件版本不匹配: 期望 " << kVersion
                  << ", 实际 " << version << std::endl;
        fclose(fp);
        return false;
    }

    uint64_t count = 0;
    fread(&count, sizeof(count), 1, fp);

    chunks_.clear();
    chunks_.reserve(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
        chunks_.push_back(Chunk::deserialize(fp));
    }

    bool ok = (ferror(fp) == 0);
    fclose(fp);
    return ok;
}
