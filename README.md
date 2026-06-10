# MRAG — 基于 llama.cpp 的本地 RAG 小说问答系统

# 项目介绍

这是一个跑在本地的 RAG（检索增强生成）系统——把一本小说喂给它，然后使用自然语言问问题，本地模型会去小说里找到相关内容后回答你。

# 目录结构

```
大作业/
├── config.json          ← 所有可调参数都在这一个文件里
├── CMakeLists.txt       ← CMake 构建脚本
├── download_models.py   ← 一键下载模型 + 四大名著 + llama.cpp 源码
├── requirements.txt     ← Python 依赖（huggingface-hub）
├── src/                 ← 所有源码
│   ├── chunk.h / .cc            → Chunk 结构体 + 二进制读写
│   ├── config.h / .cc           → 配置结构体 + JSON 解析
│   ├── document.h / .cc         → 文档切分（章节识别、UTF-8 安全切割）
│   ├── vectordatabase.h / .cc   → 向量存储 + 相似度检索 + 磁盘持久化
│   ├── llamamodel.h / .cc       → llama.cpp 模型基类（RAII 封装）
│   ├── embeddingengine.h / .cc  → 文本 → Embedding 向量
│   ├── generationengine.h / .cc → RAG 提示词构造 + 流式生成
│   └── main.cc                  → 入口：解析命令行、交互式模型切换、调度全流程
├── patches/             ← 跨平台兼容补丁
│   └── fs_path_shim.h   → 补全 std::filesystem::path 的 operator+（Linux 需要）
├── data/                ← 输入数据和输出数据库
│   ├── 三国演义.txt
│   ├── 水浒传.txt
│   ├── 红楼梦.txt
│   └── 西游记.txt
├── models/
│   ├── bge-small-zh-v1.5-f16.gguf          ← Embedding 模型
│   ├── qwen2.5-1.5b-instruct-q4_k_m.gguf   ← 1.5B 生成模型（默认）
│   └── qwen2.5-7b-instruct-q4_k_m.gguf     ← 7B 生成模型（可选热切换）
├── resources/
│   └── llama.cpp-master/
└── build/
```

- **`data/`**：输入数据（TXT 小说文本）和建库产出的 `.db` 数据库文件都放这。建库时往这写，问答时从这读。当前放了四大名著作为测试数据。
- **`models/`**：GGUF 模型文件。Embedding 模型固定是一个，生成模型默认使用 1.5B；7B 作为高配可选模型保留在目录中，但是否能在运行时热切换成功取决于设备内存/显存余量。
- **`resources/`**：第三方库源码。目前只有 llama.cpp，CMake 构建时直接把它当子目录拉进来一起编译。

# 环境要求

- **系统**：Windows / Linux（两个平台都测过）
- **编译器**：支持 C++17（MSVC 2019+ 或 GCC 9+ / Clang 10+）
- **CMake**：3.16 或更高
- **内存**：默认 1.5B 生成模型建议 8 GB 以上内存；7B 生成模型约 4\~5 GB 文件体积，实际加载和切换会产生额外内存峰值，低内存设备上可能无法运行时切换。

> **Linux 编译注意**：llama.cpp 源码在 `common/common.cpp` 等文件中使用了 `fs_path + "string"` 写法，这是 MSVC/STL 的私有扩展（非 C++ 标准）。GCC 的 libstdc++ 严格遵循标准，会导致编译报错：
>
> ```
> error: no match for 'operator+' (operand types are 'std::filesystem::path' and 'const char [2]')
> ```
>
> 项目已在 CMakeLists.txt 中配置了 `patches/fs_path_shim.h`，Linux 编译时会通过 `-include` 自动注入补全该缺失操作符，无需手动干预。

# 模型下载

需要三个 GGUF 格式的模型文件，都放到 `models/` 目录下：

| 文件                                  | 用途               | 显存        |
| ----------------------------------- | ---------------- | --------- |
| `bge-small-zh-v1.5-f16.gguf`        | Embedding（文本→向量） | \~200 MB  |
| `qwen2.5-1.5b-instruct-q4_k_m.gguf` | 生成（回答问题）         | \~1 GB    |
| `qwen2.5-7b-instruct-q4_k_m.gguf`   | 生成（更大更强，可选）      | \~4\~5 GB |

有四种下载方式，按顺手程度选一种就行。

### 方式一：Python 脚本一键下载（推荐）

项目自带了 `download_models.py`，用 HuggingFace Hub 官方接口下载，支持断点续传：

```bash
# 先装依赖
pip install -r requirements.txt

# 全部下载 三个模型 + 四大名著 + llama.cpp 源码
python download_models.py
```

### 方式二：wget 直链下载

用 HuggingFace 的直链，不需要登录：

```bash
# Embedding 模型
wget -P ./models/ https://huggingface.co/BAAI/bge-small-zh-v1.5/resolve/main/bge-small-zh-v1.5-f16.gguf

# 1.5B 生成模型
wget -P ./models/ https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

# 7B 生成模型（可选）
wget -P ./models/ https://huggingface.co/paultimothymooney/Qwen2.5-7B-Instruct-Q4_K_M-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf
```

> `-P ./models/` 表示下载到 `models/` 目录。Windows 没有 wget 的话可以用 `curl -L -o` 替代。

### 方式三：curl 下载（Windows / Linux 通用）

```powershell
# Windows PowerShell
curl -L -o ./models/bge-small-zh-v1.5-f16.gguf https://www.modelscope.cn/models/BAAI/bge-small-zh-v1.5/resolve/master/bge-small-zh-v1.5-f16.gguf
curl -L -o ./models/qwen2.5-1.5b-instruct-q4_k_m.gguf https://www.modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/master/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

### 方式四：手动浏览器下载

去 Hugging Face 或 ModelScope 网页上搜模型名，找到 `.gguf` 文件下载后拖到 `models/` 文件夹里：

- Embedding：搜 `BAAI/bge-small-zh-v1.5`，找 `bge-small-zh-v1.5-f16.gguf`
- 1.5B：搜 `Qwen/Qwen2.5-1.5B-Instruct-GGUF`，找 `qwen2.5-1.5b-instruct-q4_k_m.gguf`
- 7B：搜 `paultimothymooney/Qwen2.5-7B-Instruct-Q4_K_M-GGUF`，找 `qwen2.5-7b-instruct-q4_k_m.gguf`

模型放到 `models/` 下即可。项目默认启动使用 1.5B（`config.json` 中固定配置为 `qwen2.5-1.5b-instruct-q4_k_m.gguf`）。7B 是高配可选项，用于设备内存充足时测试，不作为默认启动模型。

### 四大名著数据下载

测试用的小说 TXT 来自 [tennessine/corpus](https://github.com/tennessine/corpus)，UTF-8 编码、带章节标题。

**Python 脚本**（上面 `python download_models.py` 默认就会一起下载）：

```bash
python download_models.py                 # 全部下载
python download_models.py --novels-only   # 只下载四大名著
```

**wget 直链**：

```bash
wget -P ./data/ https://raw.githubusercontent.com/tennessine/corpus/main/三国演义.txt
wget -P ./data/ https://raw.githubusercontent.com/tennessine/corpus/main/水浒传.txt
wget -P ./data/ https://raw.githubusercontent.com/tennessine/corpus/main/红楼梦.txt
wget -P ./data/ https://raw.githubusercontent.com/tennessine/corpus/main/西游记.txt
```

### llama.cpp 源码下载

项目依赖 llama.cpp C API，需要将源码放在 `resources/llama.cpp-master/` 下。来源：[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)。

**Python 脚本**（上面 `python download_models.py` 默认就会一起下载并解压）：

```bash
python download_models.py                 # 全部下载
python download_models.py --skip-llama    # 跳过 llama.cpp
```

**wget 直链**：

```bash
wget -P ./resources/ https://github.com/ggerganov/llama.cpp/archive/refs/heads/master.zip
# 手动解压：将 master.zip 解压到 resources/，重命名为 llama.cpp-master
```

# 编译

```bash
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)   # Linux
cmake --build . --config Release  # Windows
```

编译完会在 `build/` 下生成 `mrag.exe`（Windows）或 `mrag`（Linux）。

# 使用方法

### 第一步：建库

先把小说喂进去，生成向量数据库。以下命令全部在**项目根目录**下执行：

```bash
# 单本逐条添加（推荐）
cd ..
./build/mrag --ingest ./data/三国演义.txt ./data/sanguo.db
./build/mrag --ingest ./data/水浒传.txt ./data/shuihu.db
./build/mrag --ingest ./data/西游记.txt ./data/xiyou.db
./build/mrag --ingest ./data/红楼梦.txt ./data/honglou.db

# 多文档一次性建库（搜索范围更大，但是搜索准度下降）
cd ..
./build/mrag --ingest ./data/三国演义.txt ./data/水浒传.txt ./data/西游记.txt ./data/红楼梦.txt ./data/four_classics.db

```

这个过程会：

1. 读 TXT 文件，识别章节标题，切成 Chunk
2. 每个 Chunk 用 Embedding 模型生成一个向量
3. 把所有 Chunk 和向量存成 .db 文件

控制台会输出进度，大概像这样：

```
[INFO] 正在处理文档: ./data/三国演义.txt
[INFO] 文档切分完成，共 128 个片段
[INFO] 正在加载 Embedding 模型: ./models/bge-small-zh-v1.5-f16.gguf
[INFO] 正在生成向量嵌入...
[INFO] 嵌入进度: 128/128
[INFO] 正在写入向量数据库...
[INFO] 知识库构建完成，保存至: ./data/sanguo.db
[INFO] 数据库包含 128 个向量条目
```

### 第二步：问答

加载建好的数据库，进入交互问答模式（同样在项目根目录下运行）：

```bash
./build/mrag --chat ./data/sanguo.db
./build/mrag --chat ./data/shuihu.db
./build/mrag --chat ./data/xiyou.db
./build/mrag --chat ./data/honglou.db

#全 .txt 问答
# ./build/mrag --chat ./data/four_classics.db
```

然后就可以输入问题了：
（例如执行 `./build/mrag --chat ./data/sanguo.db` 之后）

```
> 刘备的字是什么？
[检索到 3 个相关片段]
  [1] 第一回: 榜文行到涿县，引出涿县中一个英雄。那人不甚好读...
  [2] 第一回: 玄德曰："我本汉室宗亲，姓刘，名备，字玄德...
  [3] 第二回: ...

根据参考上下文，刘备字玄德...
```

交互模式支持两个控制命令：

- `/reload`：交互式切换生成模型——自动扫描 `models/` 目录列出生成模型候选，按编号选择即可，**不需要手动改 config.json**
- `/exit`：退出问答

### 运行时热切换模型

默认推荐使用 1.5B 生成模型启动和验收：

```json
"generation": "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
```

输入 `/reload` 后，程序会扫描 `./models/` 目录，列出可用的生成模型：

```
--- 可用的生成模型 ---
  [1] qwen2.5-1.5b-instruct-q4_k_m.gguf
  [2] qwen2.5-7b-instruct-q4_k_m.gguf
------------------------
请选择生成模型编号（回车跳过不改）:
```

输入编号可以尝试切换生成模型，回车则保持当前模型不变。

Embedding 模型和已建好的数据库向量维度绑定，交互式 `/reload` 默认不切换 Embedding。

**注意**：7B 模型能否运行时切换成功与设备内存/显存有关。它的文件约 4\~5 GB，加载时还会有额外上下文和临时内存开销；如果机器资源不足，应继续使用默认 1.5B。项目设计上把 1.5B 作为稳定默认路径，把 7B 作为高配机器上的可选测试项。

### 单次查询

不想进交互模式的话，也可以用 `--query` 一次问完：

```bash
./build/mrag --query ./data/sanguo.db "关羽的武器是什么？"
```

# 配置文件说明

`config.json` 里的可以调的参数：

```json
{
    "model": {
        "embedding": "./models/bge-small-zh-v1.5-f16.gguf",
        "generation": "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "n_gpu_layers": 99
    },
    "document": {
        "chunk_size": 650,
        "overlap_size": 100
    },
    "retrieval": {
        "top_k": 3
    },
    "generation": {
        "n_ctx": 4096,
        "n_batch": 4096,
        "n_ubatch": 512,
        "max_output_tokens": 512,
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9
    },
    "embedding": {
        "n_ctx": 512,
        "n_batch": 512,
        "n_ubatch": 512
    }
}
```

大部分参数保持默认就行。有几个想调的可能列出在下：

- **`chunk_size`**：切多长的文本块，当前默认 650 字节。调小一点检索更精确但 Chunk 数量会变多，调太大则容易引入噪声。
- **`overlap_size`**：相邻 Chunk 重叠多少，当前默认 100 字节。保证不会因为切割把一句话劈成两半。
- **`top_k`**：检索返回几个相关 Chunk。默认 3，多了可能让 prompt 太长塞不进上下文窗口。
- **`temperature`**：生成随机性。0 最确定，1 比较放飞，默认 0.7。
- **`n_gpu_layers`**：GPU 加速层数。有 NVIDIA 显卡就设大一点（比如 99），只用 CPU 就设 0。

# 已知问题

1. 模型只支持 GGUF 格式，其他格式不行。
2. 输入文件只支持 UTF-8 编码的纯文本 TXT。
3. 章节识别只覆盖了中文"第X章/Y回/Z卷"和英文"Chapter N"格式。
4. 如果 prompt 太长超出模型上下文窗口（比如检索片段太多），会跳过本次推理不做回答。
5. 热切换到不同维度的 Embedding 模型时，旧数据库向量可能无法匹配，需要重新 `--ingest` 建库。

# 修改记录

### 混合检索（2026-05-27）

原版纯向量检索返回的 chunk 全部来自无关章节，相似度近乎随机。已加入**关键词+向量混合检索**：

- **关键词评分**：去掉停用词（"的是在了"等），计算"内容字符密度"（density² × 0.6 + bigram × 0.4）
- **混合公式**：基础分为 `0.45 × cosSimilarity + 0.55 × keywordScore`，短事实题会额外加入主语/属性证据加权
- **UTF-8 控制台**：`SetConsoleCP(CP_UTF8)` 确保 Windows 终端输入正确编码

**效果对比**：

| 查询      | 修复前检索结果     | 修复后检索结果           |
| ------- | ----------- | ----------------- |
| 桃园结义在哪  | 第32/23/20回  | 第5/29/**1回** 章节对了 |
| 十常侍是哪些人 | 第32/??/20回  | **第2/2/1回** 命中相关章 |

- 检索到了**正确章节**，但 LLM 因上下文不够完整仍出现幻觉
- 例如：桃园结义→答"平原县"（正确是"桃园"）；十常侍→答"窦武、陈蕃…"（全是编造的）

### 多文档建库（2026-06-01）

原版 `--ingest` 一次只能处理一本 TXT。现在已支持一次传入多个 TXT 文件，并将所有 chunk 写入同一个向量数据库：

```bash
./build/mrag --ingest ./data/三国演义.txt ./data/水浒传.txt ./data/西游记.txt ./data/红楼梦.txt ./data/four_classics.db
```

- 命令行规则：最后一个参数是输出 `.db` 路径，前面的参数全部视为 TXT 输入文件
- 每个文档会依次完成章节识别、chunk 切分和 embedding 生成
- 适合用四大名著作为 5.2 加分项的多文档建库测试对象

### 配置文件热切换（2026-06-05）

原版 `--chat` 启动后模型路径固定，切换模型必须退出重启。现在聊天模式会重新读取 `config.json`，支持运行时切换模型路径：

- 新增 `/reload` 命令：在不退出 `--chat` 的情况下重新加载配置
- 自动检测：每轮用户提问前也会检查配置变化
- 支持从 `./models/qwen2.5-1.5b-instruct-q4_k_m.gguf` 切换到 `./models/qwen2.5-7b-instruct-q4_k_m.gguf`
- 当前版本默认 1.5B 稳定启动；7B 热切换会尝试先释放旧生成模型以降低内存峰值，但是否成功仍取决于设备资源

### 交互式 /reload 选模型（2026-06-07）

把热切换从"改 config.json → `/reload` 读取"改为"直接扫描 `models/` 目录 → 按编号选择"：

- 输入 `/reload` 后自动扫描 `./models/` 下所有 `.gguf` 文件，列出编号列表
- 输入编号即可尝试切换生成模型；Embedding 模型与数据库向量绑定，交互式 `/reload` 默认不切换 Embedding
- 不再需要手动编辑 config.json，整个切换过程在交互界面内完成
- `FindFirstFile`/`FindNextFile`（Windows）/ `opendir`（Linux）扫描目录、`std::stoul` 解析编号；7B 加载失败时会提示资源约束，推荐继续使用默认 1.5B

### 一键下载脚本 + requirements.txt（2026-06-07）

原来的模型下载依赖用户自己去网页找链接，现在提供四种方式：

- **Python 脚本**：项目自带了 `download_models.py`，用 `huggingface_hub` 官方接口下载全部三个模型，支持断点续传、支持 `--skip-7b` / `--skip-novels` / `--novels-only` 等选项。同时支持从 [tennessine/corpus](https://github.com/tennessine/corpus) 下载四大名著 TXT、从 [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) 下载并自动解压 llama.cpp 源码
- **wget**：模型用 HuggingFace 直链、四大名著用 raw\.githubusercontent.com 直链、llama.cpp 用 GitHub archive 直链，不需要登录
- **curl**：Windows 用户没有 wget 时的替代方案
- **手动浏览器**：保留原有方式，适合网络受限的环境

`requirements.txt` 列出了 Python 依赖（目前只有 `huggingface-hub>=0.20.0`），`pip install -r requirements.txt` 一条命令装好。四大名著下载不需要额外依赖（用的标准库 `urllib`）。

### Linux 编译兼容性补丁 fs\_path\_shim.h（2026-06-07）

llama.cpp 源码中存在跨平台兼容性 bug —— `common/common.cpp` 等文件使用了 `fs_path + "string"` 的写法。`operator+` 不是 C++ 标准的 `std::filesystem::path` 成员，是 MSVC/STL 的私有扩展，GCC 的 libstdc++ 严格遵循标准因此编译失败。

解决方案：`patches/fs_path_shim.h` 提供了一个标准兼容的 `operator+` 实现（内部用 `operator+=`），CMakeLists.txt 在非 MSVC 环境下通过 `-include` 自动注入。

### 问答稳定性与指令处理修复（2026-06-07）

实测 `--chat` 中出现了三个交互问题：检索片段打印后没有回答、交互式切换模型后下一轮又触发配置热切换、以及错拼 `/relaod` 会被当作普通问题检索。已完成以下修复：

- **生成阶段崩溃修复**：自回归生成时不再对 `llama_batch_get_one()` 返回的临时 batch 调用 `llama_batch_free()`；改为每个新 token 使用 `llama_batch_init(1, 0, 1)` 正规分配 batch，填入 `token/pos/seq_id/logits` 后 decode，并在 decode 后释放
- **空回答兜底**：如果模型采样后没有生成有效文本，会返回 `[提示] 模型没有生成有效内容，请换个问法或输入 /reload 切换模型后再试。`，避免用户看到程序静默结束
- **单轮异常保护**：`chatLoop` 中每轮问答增加 `try/catch`，某一轮加载或推理失败时会提示错误并回到 `>`，不会直接退出整个聊天循环
- **热切换配置一致性**：交互式 `/reload` 按编号选择模型后，会把当前模型路径写回 `config.json`，避免内存中已切到 7B、但下一轮自动读取旧配置又切回 1.5B
- **配置文件自动检测保留**：普通提问前仍会检查 `config.json` 是否被外部编辑；如果变化，会自动重建 Embedding/Generation 引擎
- **指令输入识别**：输入会先去除首尾空白和 UTF-8 BOM；所有 `/` 开头但不是 `/reload`、`/exit` 的内容都会提示未知指令，不再送入检索流程

验证：

```text
> /relaod
[WARN] 未知指令: /relaod
[INFO] 可用指令: /reload, /exit
```

```text
./build/mrag --query ./data/sanguo.db "刘备的字是什么？"
[INFO] 正在生成回答...
刘备的字是玄德。
```

### 默认 1.5B 启动与 7B 资源约束说明（2026-06-10）

实测 7B 模型文件虽然可以放入 `models/`，但运行时从 1.5B 切换到 7B 是否成功，与设备内存、显存、内存碎片和 llama.cpp 后端分配策略有关。低内存设备上可能出现模型加载阶段被系统结束的情况，这不是问答逻辑错误，而是资源约束。

因此当前设计明确调整为：

- `config.json` 默认始终使用 `./models/qwen2.5-1.5b-instruct-q4_k_m.gguf`
- 1.5B 是稳定启动、课程验收和普通设备测试的默认路径
- 7B 作为高配可选模型保留，可用于设备资源充足时测试效果上限
- README 和 DESIGN 前置说明了 7B 热切换与设备内存/性能相关，避免把高配模型加载失败误判为程序功能失败
- 交互式 `/reload` 默认只切换生成模型；Embedding 模型与数据库向量绑定，不建议在已有数据库上随意切换

这个取舍保证基础问答流程、配置默认值、建库和普通查询都走稳定的 1.5B 路径，同时保留 7B 扩展能力。

### 短事实题检索增强与保守拒答（2026-06-10）

实测当 `chunk_size=1000, overlap_size=180` 时，问题“刘备的字是什么”会检索到后文含“玄德曰”“备愿”等片段，却没有命中第一回“姓刘名备，字玄德”，导致模型从错误上下文里猜出“孔明”或“玄”。

本次修复不追求所有题都答对，而是减少“检索错了还硬答”的情况：

- `config.json` 文档切分回调到更稳的 `chunk_size=650, overlap_size=100`
- 对“是什么/是谁/哪里/哪些/多少”等短事实题增加专门检索加权
- 从“刘备的字是什么”这类问题中提取主语“刘备”，并识别原文里的“姓刘名备”这种非连续写法
- 如果短事实题的候选片段不包含主语证据，会降低排序分数
- 对“字/名/姓/号/武器”等属性词增加证据加权，让含有明确原文模式的片段更容易进 top-K
- Prompt 改为保守回答：上下文没有直接证据、人物不匹配或只是相似名字时，必须回答“根据当前片段无法得到答案”，避免继续编造

这个修改的目标是提升检索和拒答质量，而不是保证每个问题都能被命中。总结类、跨章节类问题仍然属于 RAG 的弱项。
