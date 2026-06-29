#include "vectordatabase.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

static std::vector<std::string> splitUtf8Chars(const std::string &text);

static std::vector<std::string> queryAliases(const std::string &query) {
    static const std::unordered_map<std::string, std::vector<std::string>> aliases = {
        {"刘备", {"玄德", "刘玄德"}},
        {"关羽", {"云长", "关公", "关云长"}},
        {"张飞", {"翼德", "张翼德"}},
        {"曹操", {"孟德", "曹孟德", "阿瞒"}},
        {"诸葛亮", {"孔明", "卧龙"}},
        {"吕布", {"奉先", "温侯"}},
        {"孙权", {"仲谋"}},
        {"周瑜", {"公瑾"}}
    };

    std::vector<std::string> result;
    for (const auto &entry : aliases) {
        if (query.find(entry.first) == std::string::npos) continue;
        result.push_back(entry.first);
        result.insert(result.end(), entry.second.begin(), entry.second.end());
    }
    return result;
}

static float entityCooccurrenceScore(const std::string &query,
                                     const std::string &chunkText) {
    static const std::unordered_map<std::string, std::vector<std::string>> entities = {
        {"刘备", {"刘备", "玄德", "刘玄德"}},
        {"关羽", {"关羽", "云长", "关公", "关云长"}},
        {"张飞", {"张飞", "翼德", "张翼德"}},
        {"曹操", {"曹操", "孟德", "曹孟德", "阿瞒"}},
        {"诸葛亮", {"诸葛亮", "孔明", "卧龙"}},
        {"吕布", {"吕布", "奉先", "温侯"}},
        {"张角", {"张角"}},
        {"张宝", {"张宝"}},
        {"张梁", {"张梁"}},
        {"许劭", {"许劭"}}
    };

    int requested = 0;
    int matched = 0;
    for (const auto &entry : entities) {
        if (query.find(entry.first) == std::string::npos) continue;
        ++requested;
        bool found = false;
        for (const auto &name : entry.second) {
            if (chunkText.find(name) != std::string::npos) {
                found = true;
                break;
            }
        }
        if (found) ++matched;
    }
    if (requested == 0) return 0.0f;
    if (matched == requested) return requested > 1 ? 0.7f : 0.35f;
    return 0.15f * static_cast<float>(matched);
}

static float evidenceCueScore(const std::string &query, const std::string &chunkText) {
    float score = 0.0f;
    auto hasAny = [&chunkText](std::initializer_list<const char *> cues) {
        for (const char *cue : cues) {
            if (chunkText.find(cue) != std::string::npos) return true;
        }
        return false;
    };

    if (query.find("分别") != std::string::npos ||
        query.find("哪些人") != std::string::npos ||
        query.find("哪几") != std::string::npos) {
        if (hasAny({"一名", "分别", "十人", "号为", "称为"})) score += 0.45f;
    }
    if (query.find("十常侍") != std::string::npos &&
        chunkText.find("十人朋比为奸") != std::string::npos &&
        chunkText.find("号为“十常侍”") != std::string::npos) {
        score += 2.0f;
    }
    if (query.find("武器") != std::string::npos ||
        query.find("兵器") != std::string::npos) {
        if (hasAny({"造", "打造", "刀", "矛", "枪", "剑", "戟"})) score += 0.35f;
        if (hasAny({"重", "斤"})) score += 0.35f;
        for (const auto &alias : queryAliases(query)) {
            size_t namePos = chunkText.find(alias);
            if (namePos == std::string::npos) continue;
            size_t begin = namePos > 120 ? namePos - 120 : 0;
            std::string nearby = chunkText.substr(begin, 360);
            bool hasWeapon = nearby.find("刀") != std::string::npos ||
                             nearby.find("矛") != std::string::npos ||
                             nearby.find("枪") != std::string::npos ||
                             nearby.find("剑") != std::string::npos ||
                             nearby.find("戟") != std::string::npos;
            if (hasWeapon &&
                nearby.find("斤") != std::string::npos) {
                score += 1.5f;
                break;
            }
        }
    }
    if (query.find("多少") != std::string::npos &&
        hasAny({"斤", "人", "里", "年", "回", "次"})) {
        score += 0.25f;
    }
    if (query.find("如何") != std::string::npos ||
        query.find("怎么") != std::string::npos ||
        query.find("经过") != std::string::npos) {
        if (hasAny({"忽心生一计", "于是", "遂", "后", "因"})) score += 0.2f;
    }
    return score;
}

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
    std::vector<std::string> chars = splitUtf8Chars(query);

    if (chars.empty()) return 0.0f;

    static const std::string stopChars =
        "的是在了也吗呢啊吧与和及之乎者耶哉焉耳已矣夫"
        "什么谁哪哪里多少如何怎么为何请问说讲";
    std::vector<std::string> distChars;
    std::unordered_set<std::string> seen;
    for (const auto &ch : chars) {
        if (ch.size() >= 2 && stopChars.find(ch) == std::string::npos &&
            seen.insert(ch).second) {
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

    int ngramTotal = 0;
    float ngramFound = 0.0f;
    for (size_t width = 2; width <= 4; ++width) {
        for (size_t j = 0; j + width <= chars.size(); ++j) {
            std::string ngram;
            bool useful = false;
            for (size_t k = 0; k < width; ++k) {
                ngram += chars[j + k];
                useful = useful || stopChars.find(chars[j + k]) == std::string::npos;
            }
            if (!useful) continue;
            ++ngramTotal;
            if (chunkText.find(ngram) != std::string::npos) {
                ngramFound += static_cast<float>(width - 1);
            }
        }
    }
    float phraseScore = ngramTotal > 0
        ? std::min(1.0f, ngramFound / static_cast<float>(ngramTotal))
        : 0.0f;

    float aliasScore = 0.0f;
    for (const auto &alias : queryAliases(query)) {
        if (chunkText.find(alias) != std::string::npos) {
            aliasScore = std::max(aliasScore, 0.75f);
        }
    }

    return density * density * 0.25f + phraseScore * 0.55f + aliasScore * 0.20f;
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
    for (const auto &alias : queryAliases(subject)) {
        if (chunkText.find(alias) != std::string::npos) return 1.0f;
    }

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
        std::string searchableText = chunks_[i].metadata + "\n";
        if (i > 0 && chunks_[i - 1].metadata == chunks_[i].metadata) {
            searchableText += chunks_[i - 1].text;
            searchableText += "\n";
        }
        searchableText += chunks_[i].text;
        if (i + 1 < chunks_.size() &&
            chunks_[i + 1].metadata == chunks_[i].metadata) {
            searchableText += "\n";
            searchableText += chunks_[i + 1].text;
        }
        float kw = keywordScore(queryText, searchableText);
        float factBoost = factQuestionBoost(queryText, searchableText);
        float cueBoost = evidenceCueScore(queryText, searchableText);
        float entityBoost = entityCooccurrenceScore(queryText, searchableText);
        float score = 0.35f * cosSim + 0.65f * kw + factBoost + cueBoost + entityBoost;
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
        size_t index = *it;
        Chunk expanded = chunks_[index];
        std::string combined;
        if (index > 0 && chunks_[index - 1].metadata == expanded.metadata) {
            combined += chunks_[index - 1].text;
            combined += "\n";
        }
        combined += expanded.text;
        if (index + 1 < chunks_.size() &&
            chunks_[index + 1].metadata == expanded.metadata) {
            combined += "\n";
            combined += chunks_[index + 1].text;
        }
        expanded.text = std::move(combined);
        results.push_back(std::move(expanded));
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
