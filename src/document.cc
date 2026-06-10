#include "document.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <regex>
#include <algorithm>
#include <cassert>

DocumentProcessor::DocumentProcessor(const AppConfig::DocumentConfig &docConfig)
    : chunkSize_(docConfig.chunk_size),
      overlapSize_(docConfig.overlap_size) {}

size_t DocumentProcessor::nextUtf8Boundary(const std::string &text, size_t pos) {
    assert(pos <= text.size());
    while (pos < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[pos]);
        if ((c & 0x80u) == 0u) return pos;
        if ((c & 0xE0u) == 0xC0u) { ++pos; return (pos < text.size()) ? pos + 1 : text.size(); }
        if ((c & 0xF0u) == 0xE0u) { pos += 2; return (pos < text.size()) ? pos + 1 : text.size(); }
        if ((c & 0xF8u) == 0xF0u) { pos += 3; return (pos < text.size()) ? pos + 1 : text.size(); }
        ++pos;
    }
    return text.size();
}

bool DocumentProcessor::isChinesePunctuation(const std::string &text, size_t pos) {
    assert(pos <= text.size());
    if (pos + 2 >= text.size()) return false;
    unsigned char c  = static_cast<unsigned char>(text[pos]);
    unsigned char b1 = static_cast<unsigned char>(text[pos + 1]);
    unsigned char b2 = static_cast<unsigned char>(text[pos + 2]);
    if (c == 0xE3 && b1 == 0x80 && b2 == 0x82) return true; // 。
    if (c == 0xEF && b1 == 0xBC && b2 == 0x81) return true; // ！
    if (c == 0xEF && b1 == 0xBC && b2 == 0x9F) return true; // ？
    if (c == 0xEF && b1 == 0xBC && b2 == 0x8C) return true; // ，
    if (c == 0xEF && b1 == 0xBC && b2 == 0x9B) return true; // ；
    if (c == 0xE3 && b1 == 0x80 && b2 == 0x81) return true; // 、
    return false;
}

bool DocumentProcessor::isChapterTitle(const std::string &line) {
    static const std::regex chapterRe(R"(^(第.*[章节回卷]|Chapter\s+\d+))");
    return std::regex_search(line, chapterRe);
}

size_t DocumentProcessor::findCutPoint(const std::string &text, size_t targetPos) const {
    assert(chunkSize_ > 0);
    if (targetPos >= text.size()) return text.size();

    size_t searchStart = (targetPos > overlapSize_) ? (targetPos - overlapSize_) : 0;
    for (size_t i = targetPos; i > searchStart && i > 0; --i) {
        if (text[i] == '\n') return i;
        if (isChinesePunctuation(text, i)) return nextUtf8Boundary(text, i + 3);
    }
    return nextUtf8Boundary(text, targetPos);
}

// 读 TXT → 逐行判断章节标题 → 按 chunk_size 切分 → findCutPoint 找断点 → overlap 衔接
std::vector<Chunk> DocumentProcessor::processNovel(const std::string &filepath) {
    assert(chunkSize_ > 0);
    assert(overlapSize_ < chunkSize_);
    assert(!filepath.empty());
    std::vector<Chunk> chunks;

    FILE *fp = fopen(filepath.c_str(), "rb");
    if (!fp) {
        throw std::runtime_error("无法打开文件: " + filepath);
    }

    std::string fullText;
    std::string currentChapter = "正文";
    char lineBuf[65536];

    while (fgets(lineBuf, sizeof(lineBuf), fp)) {
        std::string line(lineBuf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
            line.pop_back();
        std::string trimmed = line;
        trimmed.erase(0, trimmed.find_first_not_of(" \t\n\r\f\v"));
        trimmed.erase(trimmed.find_last_not_of(" \t\n\r\f\v") + 1);

        if (isChapterTitle(trimmed)) {
            auto splitChunks = [&]() {
                size_t pos = 0;
                while (pos < fullText.size()) {
                    size_t cutPos = findCutPoint(fullText, pos + chunkSize_);
                    if (cutPos <= pos) cutPos = nextUtf8Boundary(fullText, pos + chunkSize_);
                    if (cutPos > fullText.size()) cutPos = fullText.size();
                    std::string chunkText = fullText.substr(pos, cutPos - pos);
                    if (!chunkText.empty()) {
                        assert(cutPos > pos);
                        chunks.emplace_back(nextChunkId_++, chunkText, currentChapter);
                    }
                    if (cutPos >= fullText.size()) break;
                    pos = (cutPos > overlapSize_) ? (cutPos - overlapSize_) : 0;
                    pos = nextUtf8Boundary(fullText, pos);
                }
            };
            splitChunks();
            fullText.clear();
            currentChapter = trimmed;
            fullText += line + "\n";
        } else {
            fullText += line + "\n";
        }
    }

    if (!fullText.empty()) {
        size_t pos = 0;
        while (pos < fullText.size()) {
            size_t cutPos = findCutPoint(fullText, pos + chunkSize_);
            if (cutPos <= pos) cutPos = nextUtf8Boundary(fullText, pos + chunkSize_);
            if (cutPos > fullText.size()) cutPos = fullText.size();
            std::string chunkText = fullText.substr(pos, cutPos - pos);
            if (!chunkText.empty()) {
                assert(cutPos > pos);
                chunks.emplace_back(nextChunkId_++, chunkText, currentChapter);
            }
            if (cutPos >= fullText.size()) break;
            pos = (cutPos > overlapSize_) ? (cutPos - overlapSize_) : 0;
            pos = nextUtf8Boundary(fullText, pos);
        }
    }

    fclose(fp);
    return chunks;
}
