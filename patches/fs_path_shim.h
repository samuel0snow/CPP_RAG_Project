// Shim: provide operator+ for std::filesystem::path (non-standard, needed by llama.cpp)
// 
// llama.cpp 在 common/common.cpp 等文件中使用了 fs_path + "string" 的写法。
// 这个 operator+ 不是 C++ 标准，是 MSVC/STL 的私有扩展。
// GCC 的 libstdc++ 严格遵循标准，没有这个扩展，会导致编译失败。
// 本文件补全这个缺失的 operator+，让 llama.cpp 在 Linux/GCC 上能编译通过。
#pragma once

#include <filesystem>
#include <string>

static inline std::filesystem::path operator+(const std::filesystem::path &a, const std::filesystem::path &b) {
    std::filesystem::path result = a;
    result += b;
    return result;
}

static inline std::filesystem::path operator+(const std::filesystem::path &a, const std::string &b) {
    std::filesystem::path result = a;
    result += b;
    return result;
}

static inline std::filesystem::path operator+(const std::string &a, const std::filesystem::path &b) {
    std::filesystem::path result = a;
    result += b;
    return result;
}

static inline std::filesystem::path operator+(const std::filesystem::path &a, const char *b) {
    std::filesystem::path result = a;
    result += b;
    return result;
}

static inline std::filesystem::path operator+(const char *a, const std::filesystem::path &b) {
    std::filesystem::path result = a;
    result += b;
    return result;
}
