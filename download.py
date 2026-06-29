#!/usr/bin/env python3
"""
MRAG resource downloader.

Models are downloaded from Hugging Face to ./models/.
Novel TXT files are downloaded from GitHub to ./data/.
llama.cpp source is downloaded from GitHub to ./resources/.
"""

import argparse
import os
import subprocess
import sys
import urllib.request
import zipfile

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "data")
RES_DIR = os.path.join(ROOT_DIR, "resources")

LLAMA_CPP_URL = "https://github.com/ggerganov/llama.cpp/archive/refs/heads/master.zip"
LLAMA_ZIP_NAME = "llama.cpp-master.zip"

MODELS = {
    "bge-small-zh-v1.5-f16.gguf": "CompendiumLabs/bge-small-zh-v1.5-gguf",
    "qwen2.5-1.5b-instruct-q4_k_m.gguf": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    "qwen2.5-7b-instruct-q4_k_m.gguf": "paultimothymooney/Qwen2.5-7B-Instruct-Q4_K_M-GGUF",
}

NOVELS = {
    "三国演义.txt": "https://raw.githubusercontent.com/tennessine/corpus/main/%E4%B8%89%E5%9B%BD%E6%BC%94%E4%B9%89.txt",
    "水浒传.txt": "https://raw.githubusercontent.com/tennessine/corpus/main/%E6%B0%B4%E6%B5%92%E4%BC%A0.txt",
    "红楼梦.txt": "https://raw.githubusercontent.com/tennessine/corpus/main/%E7%BA%A2%E6%A5%BC%E6%A2%A6.txt",
    "西游记.txt": "https://raw.githubusercontent.com/tennessine/corpus/main/%E8%A5%BF%E6%B8%B8%E8%AE%B0.txt",
}


def download_model(filename, repo_id):
    """Download a single GGUF model with huggingface-hub, then curl fallback."""
    from huggingface_hub import hf_hub_download

    dest = os.path.join(MODELS_DIR, filename)
    if os.path.exists(dest):
        print(f"[SKIP] {filename} 已存在，跳过", flush=True)
        return True

    print(f"[DOWNLOAD] {filename} <- {repo_id}", flush=True)
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=MODELS_DIR,
        )
        print(f"[OK] {filename} -> {path}", flush=True)
        return True
    except Exception as exc:
        print(f"[FAIL] {filename}: {exc}", file=sys.stderr, flush=True)
        return download_model_with_curl(filename, repo_id, dest)


def download_model_with_curl(filename, repo_id, dest):
    """Fallback downloader that supports resume through curl -C -."""
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/{repo_id}/resolve/main/{filename}"
    temp_dest = dest + ".part"
    print(f"[FALLBACK] 使用 curl 下载: {url}", flush=True)

    try:
        result = subprocess.run(["curl", "-L", "-C", "-", "-o", temp_dest, url])
        if result.returncode != 0:
            print(f"[FAIL] curl 下载失败，返回码: {result.returncode}", file=sys.stderr, flush=True)
            return False
        os.replace(temp_dest, dest)
        print(f"[OK] {filename} -> {dest}", flush=True)
        return True
    except Exception as exc:
        print(f"[FAIL] curl fallback 异常: {exc}", file=sys.stderr, flush=True)
        return False


def download_novel(name, url):
    """Download a single TXT file."""
    dest = os.path.join(DATA_DIR, name)
    if os.path.exists(dest):
        print(f"[SKIP] {name} 已存在，跳过")
        return True

    print(f"[DOWNLOAD] {name} <- {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[OK] {name} -> {dest}")
        return True
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}", file=sys.stderr)
        return False


def download_llama_cpp():
    """Download and extract llama.cpp master.zip."""
    os.makedirs(RES_DIR, exist_ok=True)
    zip_path = os.path.join(RES_DIR, LLAMA_ZIP_NAME)
    extract_dir = os.path.join(RES_DIR, "llama.cpp-master")

    if os.path.exists(extract_dir):
        print("[SKIP] llama.cpp-master/ 已存在，跳过")
        return True

    print(f"[DOWNLOAD] llama.cpp master.zip <- {LLAMA_CPP_URL}")
    try:
        urllib.request.urlretrieve(LLAMA_CPP_URL, zip_path)
        print(f"[OK] 下载完成: {zip_path}")
        print(f"[UNZIP] 解压到 {extract_dir} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RES_DIR)
        print("[OK] 解压完成")
        os.remove(zip_path)
        print(f"[OK] 已清理 {LLAMA_ZIP_NAME}")
        return True
    except Exception as exc:
        print(f"[FAIL] llama.cpp: {exc}", file=sys.stderr)
        return False


def download_models(args):
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 50)
    print("MRAG 模型下载")
    print(f"目标目录: {MODELS_DIR}")
    print("=" * 50)

    ok = True
    ok = download_model("bge-small-zh-v1.5-f16.gguf", MODELS["bge-small-zh-v1.5-f16.gguf"]) and ok

    if args.emb_only:
        print("\n[OK] Embedding 模型下载完成" if ok else "\n[FAIL] Embedding 模型下载失败")
        return ok

    ok = download_model("qwen2.5-1.5b-instruct-q4_k_m.gguf", MODELS["qwen2.5-1.5b-instruct-q4_k_m.gguf"]) and ok
    if args.skip_7b:
        print("[SKIP] 已跳过 7B 模型下载")
    else:
        ok = download_model("qwen2.5-7b-instruct-q4_k_m.gguf", MODELS["qwen2.5-7b-instruct-q4_k_m.gguf"]) and ok

    print("\n[OK] 模型下载完成" if ok else "\n[FAIL] 部分模型下载失败")
    print(f"models/ 文件列表: {os.listdir(MODELS_DIR)}")
    return ok


def download_novels():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("\n" + "=" * 50)
    print("四大名著数据下载")
    print("来源: https://github.com/tennessine/corpus")
    print(f"目标目录: {DATA_DIR}")
    print("=" * 50)

    ok = True
    for name, url in NOVELS.items():
        ok = download_novel(name, url) and ok

    print("\n[OK] 四大名著下载完成" if ok else "\n[FAIL] 部分四大名著下载失败")
    print(f"data/ 文件列表: {os.listdir(DATA_DIR)}")
    return ok


def parse_args():
    parser = argparse.ArgumentParser(description="下载 MRAG 所需的 GGUF 模型、四大名著 TXT 和 llama.cpp 源码")
    parser.add_argument("--hf-endpoint", default="", help="自定义 Hugging Face endpoint，例如 https://hf-mirror.com")
    parser.add_argument("--skip-7b", action="store_true", help="跳过 7B 模型下载")
    parser.add_argument("--emb-only", action="store_true", help="仅下载 Embedding 模型")
    parser.add_argument("--skip-novels", action="store_true", help="跳过四大名著下载")
    parser.add_argument("--novels-only", action="store_true", help="仅下载四大名著，不下载模型")
    parser.add_argument("--skip-llama", action="store_true", help="跳过 llama.cpp 源码下载")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    ok = True
    if not args.novels_only:
        ok = download_models(args) and ok

    if not args.skip_novels:
        ok = download_novels() and ok

    if not args.skip_llama:
        print("\n" + "=" * 50)
        print("llama.cpp 源码下载")
        print("来源: https://github.com/ggerganov/llama.cpp")
        print(f"目标目录: {RES_DIR}")
        print("=" * 50)
        ok = download_llama_cpp() and ok
        if ok:
            print("\n[OK] llama.cpp 源码下载完成")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
