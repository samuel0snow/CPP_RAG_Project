# MRAG 本地小说问答系统

项目实现了一个本地运行的 RAG 问答程序：先把小说文本切成片段并生成向量数据库，之后用户提问时，程序会先检索相关片段，再把片段和问题一起交给本地大模型生成回答。

项目主要基于 `llama.cpp`，模型文件使用 GGUF 格式。默认生成模型是 Qwen2.5-1.5B，也可以在运行时切换到其他兼容 GGUF 模型。

## 实际编译测试注意

这个项目在 Windows 和 Linux 下都做过编译测试。Linux 下遇到过一个比较具体的问题：`llama.cpp` 源码里的 `common/common.cpp` 等文件使用了 `fs_path + "string"` 这种写法，这在 MSVC/STL 下能通过，但不是标准 C++ 写法。GCC 的 libstdc++ 会报错：

```text
error: no match for 'operator+' (operand types are 'std::filesystem::path' and 'const char [2]')
```

所以项目里加了 `patches/fs_path_shim.h`。CMake 在非 MSVC 编译器下会通过 `-include` 自动注入这个头文件，补上需要的 `operator+`，不需要手动修改 `llama.cpp` 源码。

## 目录说明

```text
大作业/
├── CMakeLists.txt
├── config.json
├── download.py
├── requirements.txt
├── README.md
├── DESIGN.md
├── src/
│   ├── main.cc
│   ├── mragapp.h / mragapp.cc
│   ├── config.h / config.cc
│   ├── chunk.h / chunk.cc
│   ├── document.h / document.cc
│   ├── vectordatabase.h / vectordatabase.cc
│   ├── llamamodel.h / llamamodel.cc
│   ├── embeddingengine.h / embeddingengine.cc
│   └── generationengine.h / generationengine.cc
├── data/
├── models/
├── patches/
└── resources/
```

几个主要目录：

- `src/`：项目源码。
- `data/`：小说 TXT 和生成出来的 `.db` 向量数据库。
- `models/`：GGUF 模型文件。
- `resources/llama.cpp-master/`：第三方库 `llama.cpp` 源码。
- `patches/`：Linux 编译时用到的兼容补丁。

## 环境要求

- C++17 编译器
- CMake 3.16 或以上
- Python 3，用来运行下载脚本
- Windows 或 Linux

如果使用默认的 1.5B 生成模型，普通电脑也能跑，只是 CPU 推理会比较慢。7B 模型文件更大，运行时占用内存也明显更多，不保证每台机器都能热切换成功。

## 下载模型和数据

项目至少需要一个 Embedding 模型和一个生成模型。当前测试目录包含：

```text
bge-small-zh-v1.5-f16.gguf
bge-m3-q4_k_m.gguf
qwen2.5-1.5b-instruct-q4_k_m.gguf
qwen2.5-7b-instruct-q4_k_m.gguf
OpenAI-gpt-oss-20B-GPT5.1-5.2-DISTILL-Heretic-Uncensored-MXFP4.i1-IQ1_M.gguf
```

其中：

- `bge-small-zh-v1.5-f16.gguf` 用来生成文本向量。
- `bge-m3-q4_k_m.gguf` 是可切换的 Embedding 模型。
- `qwen2.5-1.5b-instruct-q4_k_m.gguf` 是默认问答模型。
- `qwen2.5-7b-instruct-q4_k_m.gguf` 是可选模型，不作为默认运行要求。
- OpenAI 20B 文件是约 12 GB 的高度量化生成模型。本机加载探测成功，实际速度和效果取决于硬件及量化质量。

可以用脚本下载：

```bash
pip install -r requirements.txt
python download.py
```

脚本也会下载四大名著 TXT，并下载、解压 `llama.cpp` 源码到 `resources/` 目录。

如果网络不方便，也可以手动下载模型和文本文件，只要最后目录结构和文件名对上即可。

## 编译

在项目根目录执行：

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

Windows 下生成的是 `build/mrag.exe`，Linux 下生成的是 `build/mrag`。

## 使用方法

下面的命令都在项目根目录执行。

### 1. 建库

单本文本建库：

```bash
./build/mrag --ingest ./data/三国演义.txt ./data/sanguo.db
```

也可以一次把多本书放到同一个数据库：

```bash
./build/mrag --ingest ./data/三国演义.txt ./data/水浒传.txt ./data/西游记.txt ./data/红楼梦.txt ./data/four_classics.db
```

命令规则是：最后一个参数是输出 `.db` 文件，前面的参数都是输入 TXT。

建库流程大致是：

1. 读取 TXT。
2. 按章节和长度切成 Chunk。
3. 用 Embedding 模型给每个 Chunk 生成向量。
4. 把 Chunk 和向量写入二进制 `.db` 文件。

### 2. 交互问答

```bash
./build/mrag --chat ./data/sanguo.db
```

进入交互模式后，可以直接输入问题：

```text
> 刘备的字是什么？
```

支持两个命令：

- `/reload`：扫描 `models/` 目录，分别按编号切换生成模型和 Embedding 模型。
- `/exit`：退出程序。

切换 Embedding 模型时，程序会用新模型重新计算当前数据库的全部片段向量，避免查询向量和数据库向量处于不同语义空间。重建结果只存在当前进程内存中；如需永久保存，应使用新模型重新执行 `--ingest`。

### 3. 单次查询

```bash
./build/mrag --query ./data/sanguo.db "关羽的武器是什么？"
```

这个模式适合快速测试，不进入交互循环。

## 配置文件

主要参数在 `config.json`：

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

常改的几个参数：

- `chunk_size`：每个文本片段的大致长度。
- `overlap_size`：相邻片段重叠的长度。
- `retrieval.top_k`：每次检索返回几个片段。
- `max_output_tokens`：回答最多生成多少 token。
- `n_gpu_layers`：GPU 加速层数。只用 CPU 时可以设为 0。

## 当前实现的功能

- 支持一本或多本 TXT 建库。
- 支持二进制保存和加载向量数据库。
- 支持交互问答和单次查询。
- 检索时同时使用向量相似度和简单关键词分数。
- 对“是什么、是谁、哪里、哪些”这类短事实问题做了一点额外加权。
- 支持 `/reload` 在运行时选择生成模型和 Embedding 模型。
- Windows 终端下设置 UTF-8 输入输出，避免中文乱码。

## 已知限制

1. 只支持 GGUF 模型。
2. 输入文档只处理 UTF-8 编码的 TXT。
3. 章节识别主要支持“第X章 / 第X回 / 第X卷”和英文 `Chapter N`。
4. 如果 prompt 超过模型上下文窗口，程序会跳过本次生成。
5. 多文档合并建库时，metadata 目前只记录章节名，没有额外记录书名。
6. RAG 只能根据检索到的片段回答。片段没有命中时，模型可能答不出来，或者只能返回“根据当前片段无法得到答案”。
7. 大模型是否能运行时切换成功，取决于机器内存、显存和 llama.cpp 后端分配情况。

## 测试和修改记录

这部分保留的是我测试过程中实际改过的内容。它不一定是完整开发日志，但能说明几个主要问题是怎么发现和处理的。

### 2026-05-27：混合检索

一开始只用向量相似度检索，短事实问题经常搜到无关章节。比如问“桃园结义在哪”“十常侍是哪些人”时，返回片段看起来和问题有点像，但不一定真的包含答案。

后来在 `VectorDatabase::search()` 里加了简单关键词分数，最终分数大致是：

```text
0.45 * cosine_similarity + 0.55 * keyword_score
```

后面又给“是什么 / 是谁 / 哪里 / 哪些 / 多少”这类短事实问题加了一点主语和属性词判断。这个办法不能保证每题都答对，但比纯向量检索稳定一些。

当时的测试现象：

| 问题 | 修改前 | 修改后 |
| --- | --- | --- |
| 桃园结义在哪 | 经常排到后面无关章节 | 能把第一回相关片段排上来 |
| 十常侍是哪些人 | 检索结果比较散 | 更容易命中前几回相关内容 |

这里也发现一个问题：检索到正确章节，不代表模型一定回答正确。如果上下文不够完整，生成模型仍可能猜答案，所以后面又改了 prompt，让它在证据不足时保守回答。

### 2026-06-01：多文档建库

原来 `--ingest` 一次只处理一本 TXT。为了测试四大名著合并检索，改成了最后一个参数作为输出 `.db`，前面的参数都作为输入 TXT。

示例：

```bash
./build/mrag --ingest ./data/三国演义.txt ./data/水浒传.txt ./data/西游记.txt ./data/红楼梦.txt ./data/four_classics.db
```

这样可以把多本文档写入同一个向量数据库。实际测试后也发现，多文档库的搜索范围更大，有时会降低短问题的命中率，所以单本书问答仍然更稳定。

### 2026-06-05：配置热切换

早期版本在 `--chat` 启动后，模型路径就固定了。想换生成模型必须退出程序，改 `config.json` 后再重新启动。

后来加了 `/reload`，让聊天模式可以重新读取配置并重建模型对象。普通提问前也会检查 `config.json` 是否变化，如果变化就更新运行时配置。

### 2026-06-07：交互式选择模型

继续测试时发现，每次手动改 `config.json` 还是麻烦，所以把 `/reload` 改成了交互式选择：

```text
--- 可用的生成模型 ---
  [1] qwen2.5-1.5b-instruct-q4_k_m.gguf
  [2] qwen2.5-7b-instruct-q4_k_m.gguf
------------------------
请选择生成模型编号（回车跳过不改）:
```

实现上是在 `models/` 目录里扫描 `.gguf` 文件。Windows 用 `FindFirstFileA` / `FindNextFileA`，Linux 用 `opendir` / `readdir`。这里没有用 `std::filesystem`，因为我在 MinGW 环境下遇到过链接问题。

当前版本也会显示 Embedding 模型菜单。选择后先执行模型探测，再重建当前数据库的全部向量，全部完成后才提交切换。

### 2026-06-15：mragapp 与双模型切换

按最终提交要求新增 `mragapp.h` 和 `mragapp.cc`。应用初始化、命令行解析、建库、问答及模型切换逻辑位于 `mragapp.cc`，`main.cc` 只调用 `runMragApp()`。

模型扫描将文件名包含 `bge` 或 `embed` 的 GGUF 放入 Embedding 菜单，其余放入生成模型菜单。因此新增的 OpenAI 20B GGUF 会自动出现在生成模型列表中。程序还新增 `--probe-emb`，与原有 `--probe-gen` 分别验证两类模型。

### 2026-06-07：下载脚本和依赖文件

为了减少手动准备环境的步骤，补了 `download.py` 和 `requirements.txt`。脚本主要做三件事：

- 下载 GGUF 模型。
- 下载四大名著 TXT。
- 下载并解压 `llama.cpp` 源码。

`requirements.txt` 目前主要是 `huggingface-hub`。小说文本下载使用 Python 标准库里的 `urllib`。

### 2026-06-07：Linux 编译兼容补丁

Linux 编译时遇到 `std::filesystem::path` 加字符串的报错：

```text
error: no match for 'operator+' (operand types are 'std::filesystem::path' and 'const char [2]')
```

原因是 `llama.cpp` 里用了 MSVC 能接受、但 GCC 不接受的写法。项目里加了 `patches/fs_path_shim.h`，内部用标准的 `operator+=` 实现了缺的 `operator+`。CMake 在非 MSVC 环境下自动给 `llama`、`ggml` 和 `mrag` 加上这个头文件。

### 2026-06-07：问答稳定性和指令处理

交互测试时遇到过几个问题：

- 有时打印了检索片段，但生成阶段没有正常输出。
- `/reload` 切换后，下一轮又被旧配置影响。
- 输入错拼的 `/relaod` 会被当作普通问题检索。

对应修改：

- 自回归生成阶段改成每个新 token 正规创建和释放 `llama_batch`。
- 如果模型没有生成有效内容，返回提示文本，而不是静默结束。
- `chatLoop()` 每轮问答加了 `try/catch`，单轮失败后仍能回到输入提示符。
- 交互式 `/reload` 成功后把模型路径写回 `config.json`。
- 对 `/` 开头但不是 `/reload`、`/exit` 的输入，直接提示未知指令。

测试过的输入：

```text
> /relaod
[WARN] 未知指令: /relaod
[INFO] 可用指令: /reload, /exit
```

### 2026-06-10：默认 1.5B，7B 作为可选模型

实际测试中，7B 模型文件虽然能放进 `models/`，但从 1.5B 热切换到 7B 是否成功和机器内存、显存、内存碎片都有关系。低内存环境下可能在模型加载阶段失败。

所以现在默认配置固定使用：

```json
"generation": "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
```

1.5B 作为稳定演示和验收路径，7B 只作为配置较好时的可选测试。

### 2026-06-10：短事实题检索增强和保守回答

测试“刘备的字是什么”时发现，如果 `chunk_size` 太大，检索可能命中包含“玄德曰”“备愿”等字样的后文片段，却没有命中第一回里“姓刘名备，字玄德”的直接证据。这样模型容易从错误上下文里猜。

后来做了几处调整：

- `chunk_size` 调回 650，`overlap_size` 调回 100。
- 对短事实题提取类似“刘备”这样的主语。
- 对“字 / 名 / 姓 / 号 / 武器”等属性词做额外加权。
- 如果上下文没有直接证据，prompt 要求模型回答“根据当前片段无法得到答案”。

现在短事实题比最早版本稳定，但仍然依赖检索结果。

### 2026-06-15：检索增强与重新建库

- 支持刘备/玄德、关羽/云长/关公、张飞/翼德等实体别名。
- 对多实体共同出现，以及武器、重量、名单类直接证据加权。
- 评分和返回上下文都包含同章节前后相邻 Chunk，避免答案落在切割边界外。

已使用新切块和检索代码重新生成 `data/sanguo.db`，旧库保存在 `data/sanguo_old_20260610.db`。
