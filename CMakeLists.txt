cmake_minimum_required(VERSION 3.16)
project(mrag LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "")
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "")
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "")
set(LLAMA_CURL OFF CACHE BOOL "")

add_subdirectory(resources/llama.cpp-master)

add_executable(mrag
    src/main.cc
    src/config.cc
    src/chunk.cc
    src/document.cc
    src/vectordatabase.cc
    src/llamamodel.cc
    src/embeddingengine.cc
    src/generationengine.cc
)

target_include_directories(mrag PRIVATE src)
target_include_directories(mrag PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/llama.cpp-master/include
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/llama.cpp-master/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/llama.cpp-master/ggml/src
)

target_link_libraries(mrag PRIVATE ggml llama)

# Linux/GCC 上 llama.cpp 使用了 MSVC 非标准扩展 operator+(fs::path, fs::path)
# 通过 shim 头文件补全该缺失操作符
if(NOT MSVC)
    target_compile_options(llama PRIVATE -include "${CMAKE_CURRENT_SOURCE_DIR}/patches/fs_path_shim.h")
    target_compile_options(ggml PRIVATE -include "${CMAKE_CURRENT_SOURCE_DIR}/patches/fs_path_shim.h")
    target_compile_options(mrag PRIVATE -include "${CMAKE_CURRENT_SOURCE_DIR}/patches/fs_path_shim.h")
endif()

if(MSVC)
    target_compile_options(mrag PRIVATE /W4 /utf-8)
else()
    target_compile_options(mrag PRIVATE -Wall -Wextra -pedantic -O2)
endif()
