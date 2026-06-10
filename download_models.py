#!/usr/bin/env python3
"""
MRAG 资源下载脚本
模型：从 HuggingFace 下载 GGUF 文件到 ./models/
数据：从 GitHub 下载四大名著 TXT 到 ./data/
源码：从 GitHub 下载 llama.cpp master zip 到 ./resources/
默认下载全部；支持 --skip-7b / --skip-novels / --skip-llama 等选项。
"""

import os
import sys
import argparse
import urllib.request
import zipfile

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")

# llama.cpp master 源码 zip
LLAMA_CPP_URL = "https://github.com/ggerganov/llama.cpp/archive/refs/heads/master.zip"
LLAMA_ZIP_NAME = "llama.cpp-master.zip"

MODELS = {
    "bge-small-zh-v1.5-f16.gguf": "BAAI/bge-small-zh-v1.5",
    "qwen2.5-1.5b-instruct-q4_k_m.gguf": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    "qwen2.5-7b-instruct-q4_k_m.gguf": "paultimothymooney/Qwen2.5-7B-Instruct-Q4_K_M-GGUF",
}

# 四大名著 — 来源：https://github.com/tennessine/corpus
NOVELS = {
    "三国演义.txt": "https://raw.githubusercontent.com/tennessine/corpus/main/%E4%B8%89%E5%9B%BD%E6%BC%94%E4%B9%89.txt",
    "水浒传.txt":  "https://raw.githubusercontent.com/tennessine/corpus/main/%E6%B0%B4%E6%B5%92%E4%BC%A0.txt",
    "红楼梦.txt":  "https://raw.githubusercontent.com/tennessine/corpus/main/%E7%BA%A2%E6%A5%BC%E6%A2%A6.txt",
    "西游记.txt":  "https://raw.githubusercontent.com/tennessine/corpus/main/%E8%A5%BF%E6%B8%B8%E8%AE%B0.txt",
}


def download_model(filename, repo_id):
    """通过 huggingface-hub 下载单个 GGUF 模型文件。"""
    from huggingface_hub import hf_hub_download

    dest = os.path.join(MODELS_DIR, filename)
    if os.path.exists(dest):
        print(f"[SKIP] {filename} 已存在，跳过")
        return

    print(f"[DOWNLOAD] {filename} <- {repo_id}")
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"[OK] {filename} -> {path}")
    except Exception as e:
        print(f"[FAIL] {filename}: {e}", file=sys.stderr)


def download_novel(name, url):
    """通过 urllib 下载单个 TXT 文件。"""
    dest = os.path.join(DATA_DIR, name)
    if os.path.exists(dest):
        print(f"[SKIP] {name} 已存在，跳过")
        return

    print(f"[DOWNLOAD] {name} <- {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[OK] {name} -> {dest}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}", file=sys.stderr)


def download_llama_cpp():
    """下载 llama.cpp master.zip 并自动解压到 resources/。"""
    os.makedirs(RES_DIR, exist_ok=True)
    zip_path = os.path.join(RES_DIR, LLAMA_ZIP_NAME)
    extract_dir = os.path.join(RES_DIR, "llama.cpp-master")

    if os.path.exists(extract_dir):
        print(f"[SKIP] llama.cpp-master/ 已存在，跳过")
        return

    print(f"[DOWNLOAD] llama.cpp master.zip <- {LLAMA_CPP_URL}")
    try:
        urllib.request.urlretrieve(LLAMA_CPP_URL, zip_path)
        print(f"[OK] 下载完成: {zip_path}")
        print(f"[UNZIP] 解压到 {extract_dir} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RES_DIR)
        print(f"[OK] 解压完成")
        os.remove(zip_path)
        print(f"[OK] 已清理 {LLAMA_ZIP_NAME}")
    except Exception as e:
        print(f"[FAIL] llama.cpp: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="下载 MRAG 所需的 GGUF 模型、四大名著 TXT 和 llama.cpp 源码")
    parser.add_argument("--skip-7b", action="store_true", help="跳过 7B 模型下载")
    parser.add_argument("--emb-only", action="store_true", help="仅下载 Embedding 模型")
    parser.add_argument("--skip-novels", action="store_true", help="跳过四大名著下载")
    parser.add_argument("--novels-only", action="store_true", help="仅下载四大名著，不下载模型")
    parser.add_argument("--skip-llama", action="store_true", help="跳过 llama.cpp 源码下载")
    args = parser.parse_args()

    if not args.novels_only:
        os.makedirs(MODELS_DIR, exist_ok=True)
        print("=" * 50)
        print("MRAG 模型下载")
        print(f"目标目录: {MODELS_DIR}")
        print("=" * 50)

        download_model("bge-small-zh-v1.5-f16.gguf", MODELS["bge-small-zh-v1.5-f16.gguf"])

        if args.emb_only:
            print("\n[OK] Embedding 模型下载完成")
        else:
            download_model("qwen2.5-1.5b-instruct-q4_k_m.gguf", MODELS["qwen2.5-1.5b-instruct-q4_k_m.gguf"])
            if args.skip_7b:
                print("[SKIP] 已跳过 7B 模型下载")
            else:
                download_model("qwen2.5-7b-instruct-q4_k_m.gguf", MODELS["qwen2.5-7b-instruct-q4_k_m.gguf"])
            print("\n[OK] 模型下载完成")
            print(f"models/ 文件列表: {os.listdir(MODELS_DIR)}")

    if not args.skip_novels:
        os.makedirs(DATA_DIR, exist_ok=True)
        print("\n" + "=" * 50)
        print("四大名著数据下载")
        print(f"来源: https://github.com/tennessine/corpus")
        print(f"目标目录: {DATA_DIR}")
        print("=" * 50)

        for name, url in NOVELS.items():
            download_novel(name, url)

        print("\n[OK] 四大名著下载完成")
        print(f"data/ 文件列表: {os.listdir(DATA_DIR)}")

    if not args.skip_llama:
        print("\n" + "=" * 50)
        print("llama.cpp 源码下载")
        print(f"来源: https://github.com/ggerganov/llama.cpp")
        print(f"目标目录: {RES_DIR}")
        print("=" * 50)
        download_llama_cpp()
        print("\n[OK] llama.cpp 源码下载完成")


if __name__ == "__main__":
    main()
