#pragma once

#include "chunk.h"
#include "config.h"

#include <string>
#include <vector>
#include <cstddef>

class DocumentProcessor {
public:
    explicit DocumentProcessor(const AppConfig::DocumentConfig &docConfig);

    // 读取 UTF-8 小说 TXT，按章节识别 → 切分 → overlap，返回 Chunk 列表
    std::vector<Chunk> processNovel(const std::string &filepath);

private:
    size_t chunkSize_;
    size_t overlapSize_;
    uint64_t nextChunkId_ = 0;

    // 寻找最佳切割位置：优先在句号/感叹号/问号/换行处切开
    size_t findCutPoint(const std::string &text, size_t targetPos) const;

    // 从 pos 往后找到下一个合法 UTF-8 字符边界，防止截断多字节字符
    static size_t nextUtf8Boundary(const std::string &text, size_t pos);
    // 判断 pos 处是否为中文字符标点（。！？，；、）
    static bool isChinesePunctuation(const std::string &text, size_t pos);
    // 正则匹配"第X章节回卷"或"Chapter N"格式的章节标题
    static bool isChapterTitle(const std::string &line);
};
