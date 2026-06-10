# MRAG 设计说明

## 项目目录

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

几个目录的简单定位：

- **`data/`**：放输入文件（小说 TXT）和输出的数据库文件（.db）。建库产出的二进制数据库就落在这里，问答模式也从这里加载。目前放了四大名著作为测试数据。小说来源 [tennessine/corpus](https://github.com/tennessine/corpus)，也可以通过 `python download_models.py` 自动下载。
- **`models/`**：放 GGUF 模型文件。Embedding 模型固定一个（bge-small-zh-v1.5），生成模型默认使用 1.5B。7B 作为高配可选模型保留，用于内存/显存充足的设备测试效果上限；是否能运行时切换成功取决于设备资源，不作为默认验收路径。
- **`resources/`**：放第三方库的源码，目前只有 llama.cpp。CMakeLists.txt 里通过 `add_subdirectory` 把它拉进来一起编译，不用单独装任何东西。也可以 `python download_models.py` 自动下载并解压到该目录。
- **`patches/`**：跨平台兼容性补丁。`fs_path_shim.h` 补全了 MSVC 私有的 `operator+(fs::path, fs::path)`，让 llama.cpp 在 Linux/GCC 上也能编译通过。CMakeLists.txt 在非 MSVC 环境下自动通过 `-include` 注入。

***

## 默认模型与资源约束设计

项目默认启动模型固定为 `qwen2.5-1.5b-instruct-q4_k_m.gguf`。

1.5B 模型加载快、内存压力小；

7B 模型 `qwen2.5-7b-instruct-q4_k_m.gguf` 保留为高配可选模型。它的文件体积约 4\~5 GB，实际运行时还需要上下文、KV cache、临时 batch 和系统内存余量。运行时从 1.5B 切到 7B 时，即使代码主动释放旧模型，也可能因为操作系统内存碎片、显存不足或统一内存调度失败而导致加载失败。

因此设计上不把 7B 作为默认启动模型，也**不把 7B 热切换成功视为低配设备上的必然能力**。

`/reload` 的定位是“运行时模型路径切换能力演示”：程序扫描 `models/` 下的生成模型候选，用户按编号选择。Embedding 模型默认不在交互式 `/reload` 中切换，因为数据库里的向量维度和 Embedding 模型绑定；随意切换 Embedding 会导致旧库不可用，通常需要重新 `--ingest` 建库。

***

## 整体思路

这个项目做的事说白了就两步：

**离线建库（--ingest）**：把一本或多本小说的 TXT 文件依次读进来 → 每本按章节切成 Chunk，合并到一起 → 每个 Chunk 用 Embedding 模型转成向量 → 全部存入同一个向量数据库 → 落盘成一个 .db 文件。最后一个命令行参数是输出 .db 路径，前面所有参数都是输入 TXT。

**在线问答（--chat）**：加载之前建好的 .db → 用户输入问题 → 把问题也转成 Embedding → 在数据库里搜出最相关的几个 Chunk → 把这些 Chunk 塞进提示词 → 喂给生成模型 → 流式输出回答。默认使用 1.5B 生成模型。运行期间支持 `/reload` 扫描 `models/` 目录并按编号选择生成模型，但 7B 切换是否成功受设备资源约束。也支持 `/exit` 退出。

***

## 各个模块作用

下面按依赖顺序介绍每个模块，跟上面 src/ 目录的顺序一致。这个顺序也是程序设计和实现的顺序

### 第一阶段：Chunk（chunk.h / chunk.cc）

Chunk 是整个项目里最基础的数据结构，其他所有模块都直接或间接依赖它。

```cpp
struct Chunk {
    uint64_t id;               // 唯一编号
    std::string text;          // 文本内容
    std::string metadata;      // 元信息，这里存的是"出自第几章"
    std::vector<float> embedding;  // 向量（Embedding 引擎填的）
};
```

除了字段以外，这个模块还提供了两个关键能力：

- `serialize(FILE*)`：把 Chunk 写成二进制格式。写入顺序是 id → text 长度+内容 → metadata 长度+内容 → embedding 维度+浮点数组。所有长度字段统一用 uint64\_t，保证跨平台兼容。
- `deserialize(FILE*)`：从二进制流还原一个 Chunk，顺序跟序列化一一对应。

额外加了一个 `bool empty() const`，方便后面判断一个 Chunk 是不是空的（文本和向量都没内容就算空）。

### 第二阶段：Config（config.h / config.cc）

配置模块先设计好结构体，再对接 JSON，不要反过来。这个顺序很重要——先搞清楚代码需要什么字段，再用 JSON 填值。

`AppConfig` 下面管了五个子结构体：

- **ModelConfig**：模型路径（Embedding 模型路径、生成模型路径）、GPU 层数。
- **DocumentConfig**：chunk\_size（每个 Chunk 大概多大）、overlap\_size（相邻 Chunk 之间重叠多少）。
- **RetrievalConfig**：检索时返回多少个最相似的 Chunk（top\_k）。
- **GenerationConfig**：上下文窗口大小、batch 大小、最大输出 token 数、温度、top\_k 采样、top\_p 等。
- **EmbeddingConfig**：Embedding 模型专用的上下文窗口和 batch 参数。

每个字段都有默认值。`loadConfig()` 从 config.json 读值，如果文件不存在或者某个字段缺失，就保留默认值不动，不会报错。当前版本使用手写的轻量 JSON 解析器，只解析本项目需要的字符串、整数、浮点数和对象字段，减少额外依赖。

### 第三阶段：Document（document.h / document.cc）

这个模块负责把一本小说 TXT 变成一堆 Chunk。核心入口是 `processNovel()`，内部流程是：

1. 逐行读 TXT。
2. `isChapterTitle()` 判断当前行是不是章节标题——用正则匹配"第X章"、"第X回"、"Chapter N"这类模式。
3. 识别到新章节时，把之前积攒的文本切分成 Chunk，然后清空缓冲区、更新 currentChapter。
4. 对于积攒的文本，按 chunk\_size 一批一批地切。切的时候不是硬切，而是通过 `findCutPoint()` 找最优切割位置——优先找句号（。）、感叹号（！）、问号（？）、换行符（\n），在这些自然断点处切开。
5. 每一刀切完，下一块的起始位置往前挪 overlap\_size 个字节，保证相邻 Chunk 之间有重叠，避免一句话被拦腰截断在两个 Chunk 里前后都对不上。

内部有几个关键工具函数：

- `nextUtf8Boundary()`：老师重点要求的。UTF-8 中文一个字占 3 字节，如果切割位置恰好落在某个汉字中间，就会出现乱码。这个函数负责从给定位置往后找到下一个合法的 UTF-8 字符边界。
- `isChinesePunctuation()`：判断当前字节位置是不是中文标点（。！？，；、），配合 findCutPoint 一起用。

### 第四阶段：VectorDatabase（vectordatabase.h / vectordatabase.cc）

向量数据库是检索的核心。内部就存了一个 `std::vector<Chunk> chunks_`。

**插入**：支持单条 insert 和批量 insert（通过移动语义接收 `vector<Chunk>&&`）。

**检索**：`search()` 接收一个 query 的 Embedding 向量 + 原始 query 文本，返回 top-K 个相似 Chunk。基础分数是 `0.45 × 余弦相似度 + 0.55 × 关键词分数`。此外，针对“是什么/是谁/哪里/多少”等短事实题，会额外提取主语并检查“字/名/姓/号/武器”等属性证据，含明确证据的片段加权，不含主语证据的片段降权。

实现上要求用 `priority_queue`（最小堆）而不是全排序。每次新算出一个分数就塞进堆里，大于堆顶就替换，复杂度是 O(n log k) 而不是 O(n log n)。

**持久化**：文件格式是：Magic Number（4 bytes，我用了 `0x4D524147`）→ Version（4 bytes）→ Chunk Count（8 bytes）→ 一个接一个的 Chunk 序列化数据。加载时先校验魔数和版本，不匹配就拒绝加载。

### 第五阶段：LlamaModelBase（llamamodel.h / llamamodel.cc）

这个模块是对 llama.cpp C API 的 RAII 封装，也是整个项目里最需要小心的部分。

设计上是一个基类，EmbeddingEngine 和 GenerationEngine 都继承它。构造函数根据 `ModelMode`（Embedding 还是 Generation）来设置不同的 llama\_context\_params：

- **Embedding 模式**：`embeddings = true`，后续通过 `llama_get_embeddings_seq(ctx_, 0)` 优先读取序列向量，失败时回退到 `llama_get_embeddings()`。
- **Generation 模式**：`embeddings = false`。

构造函数负责加载 GGUF 模型文件 → 创建推理上下文 → 获取 vocab。析构时必须先 `llama_free(ctx_)` 再 `llama_model_free(model_)`，顺序不能错。

`tokenize()` 是老师明确要求的方式：调用两次 `llama_tokenize`。第一次传 nullptr 拿到 token 数量，分配好 vector 后再调用第二次拿到真正的 token 列表。这样保证不会溢出。

拷贝构造、赋值、移动全部禁用，防止不小心把 model\_ 和 ctx\_ 拷出两份导致双重释放。

### 第六阶段：EmbeddingEngine（embeddingengine.h / embeddingengine.cc）

继承 LlamaModelBase，负责"文本 → 向量"。

`generateEmbedding()` 的完整步骤：

1. `sanitizeUtf8()` 清理掉残缺的 UTF-8 序列（比如被截断的多字节字符）。
2. tokenize 文本。
3. 截断到 `min(n_ctx, n_batch)`——太长会超过模型上下文窗口。
4. `llama_memory_clear()` 清空 KV Cache，避免上一次推理的缓存污染本次结果。
5. 手动构造 `llama_batch`：每个 token 设置好 token、position、sequence\_id，只给最后一个 token 开 logits。
6. `llama_decode()`（或 `llama_encode()` 如果模型有 encoder）执行推理。
7. 优先调用 `llama_get_embeddings_seq(ctx_, 0)` 拿序列 embedding；如果拿不到就回退到 `llama_get_embeddings()` 拿整个 batch 的。
8. 返回维度为 `llama_model_n_embd` 的 float 向量。

### 第七阶段：GenerationEngine（generationengine.h / generationengine.cc）

继承 LlamaModelBase，负责 RAG 问答。

内部流程分了三个部分：

**构造 Prompt**：把检索引擎返回的几个 Chunk 格式化成带出处标注的上下文字符串（类似 `[出处:第一回] 话说天下大势...`），然后用 ChatML 模板拼出完整的 prompt。system 角色写清楚"你必须且只能根据参考上下文回答，没有信息就说没有"，user 部分把上下文和问题一起贴进去。

**Prefill**：把整个 prompt tokenize → 构造 batch → `llama_decode()` 一次跑完。最后 token 开 logits，其余不开。

**自回归生成**：初始化采样器链——顺序是 top\_k → top\_p → temperature → dist（老师要求这个顺序不能改）。然后进入循环：每次 `llama_sampler_sample()` 采一个 token → `llama_vocab_is_eog()` 判断是不是结束符 → 解码成字符输出 → 构造单个 token 的 batch → `llama_decode()` → 继续循环。最多生成 `max_output_tokens` 个 token。

### 第八阶段：main.cc

入口文件，最后才写。做的事情比之前多了不少：

`BackendGuard` 是最外层的一个 RAII 结构体，构造时调 `llama_backend_init()`，析构时调 `llama_backend_free()`。放在 `main()` 的最开始，保证整个程序运行期间 backend 都是初始化好的。

三种用户运行模式：

- `--ingest <txt1> <txt2> ... <db路径>`：支持传入任意多个 TXT 文件，最后一个参数是输出 .db 路径。`buildKnowledgeBase()` 会依次对每个文档调用 `processNovel()`，把所有 Chunk 合并后统一生成 Embedding、存入数据库。比如一次性喂四大名著，四个 TXT 的 chunk 全写进同一个 four\_classics.db。
- `--chat <db路径>`：调用 `chatLoop()`，加载数据库 → 循环问答。在循环内部，用户可以输入 `/reload` 交互式选择生成模型。默认稳定路径仍是 1.5B，7B 是否能切换成功取决于设备资源。
- `--query <db路径> "问题"`：单次问答模式，不走交互循环，适合脚本调用或快速测试。

#### 交互式 /reload 选模型的实现细节

这是当前版本用于演示运行时模型路径切换的机制。用户输入 `/reload` 后，不需要退出程序、不需要手动编辑 config.json。完整流程：

1. **`interactiveReload()`**：函数入口，负责整个交互流程：
   - 用平台原生 API 扫描 `./models/` 目录（Windows：`FindFirstFileA`/`FindNextFileA`；Linux：`opendir`/`readdir`），收集所有 `.gguf` 文件名
   - 按文件名过滤出生成模型候选，避免把 bge Embedding 模型误选为生成模型
   - 显示带编号的生成模型菜单列表
   - 提示用户输入生成模型编号 → `std::stoul` 解析、校验范围
   - Embedding 模型默认不参与交互式切换，因为它和已建数据库中的向量维度绑定
   - 根据用户选择更新当前 `AppConfig`，并写回 `config.json`
2. **默认 1.5B，7B 受资源约束**：
   - `config.json` 默认固定使用 1.5B，保证普通设备和课程验收路径稳定
   - 7B 文件较大，运行时切换可能受内存、显存、内存碎片和后端分配策略影响
   - 因此 7B 是高配可选项，而不是默认启动模型或低配设备上的强保证能力
3. **引擎管理**：用 `std::unique_ptr` 而不是裸指针或栈对象。热切换的本质是在运行时释放和重建模型对象，unique\_ptr 允许通过 `reset()` 主动释放旧模型，再加载新模型，从而降低资源峰值。
4. **跨平台目录扫描**：特意避免了 `std::filesystem`，因为 MinGW 环境对 `std::filesystem` 的链接支持有问题（`codecvt` 等符号缺失）。改用 Windows API 和 POSIX API 直调，兼容性更好。Linux 侧头文件 `<dirent.h>` 条件引入。

#### Linux 跨平台兼容性补丁（patches/fs\_path\_shim.h）

llama.cpp 源码在 `common/common.cpp` 等文件中使用了 `fs_path + "string"` 写法。`operator+` 不是 C++ 标准的 `std::filesystem::path` 成员，是 MSVC/STL 的私有扩展。GCC 的 libstdc++ 严格遵循标准，因此直接编译报错：

```
error: no match for 'operator+' (operand types are 'std::filesystem::path' and 'const char [2]')
```

解决方式是 `patches/fs_path_shim.h`：用标准 `operator+=` 实现一个兼容的 `operator+`，放在全局命名空间。CMakeLists.txt 在非 MSVC 环境下通过 `-include` 编译选项自动注入给 llama target 的所有编译单元，无需修改 llama.cpp 本身一行代码。

***

## 设计上的一些取舍

**为什么当前版本使用轻量 JSON 解析器**：配置文件字段固定，当前实现只需要解析少量对象、字符串、整数和浮点数，因此用项目内的小型解析器即可覆盖需求。后续如果要完全贴合课程要求或支持更复杂 JSON，可以直接替换为 nlohmann/json，而不影响 AppConfig 的外部接口。

**为什么用 priority\_queue 而不是全排序**：数据库里可能有几百上千个 Chunk，每次问答都要检索一次。O(n log k) 比 O(n log n) 在 k 很小（比如 top\_k=3）的时候差别很大。这也是任务书上明确要求的。

**为什么 cosSim + keywordScore 双路评分**：纯余弦相似度有时候靠不住。比如用户问"张飞"，某一段虽然没有提到"张飞"两个字但语义接近可能会被排得很高，而真正提到"张飞"的那段反而排低了。加一个关键词匹配分数作为补充，效果明显更好。

**为什么 Chunk 之间要有 overlap**：如果没有重叠，一个句子可能恰好被切在两块之间，前后都对不上。有了 overlap 之后，切割点附近的文本会在前一个 Chunk 的末尾和下一个 Chunk 的开头都出现一遍，大大降低漏检的概率。

**为什么用 unique\_ptr 管理引擎而不是栈对象**：热切换需要在运行时释放和重建模型对象。如果用栈对象，无法方便地替换；用 unique\_ptr，可以通过 `reset()` 主动释放旧模型，再构造新引擎，从而降低从 1.5B 切到 7B 时的内存峰值。旧引擎释放时，析构顺序由 LlamaModelBase 保证。

**为什么多文档建库时把所有书合并到一个数据库**：建库的目的是给问答系统一块"知识土壤"，多本书合并后用户可以跨书提问（比如"比较一下刘备和宋江的性格"），不用切数据库。如果每本书单独建一个 .db，跨书问答就做不了。

***

## 已知的局限性

1. 只支持 GGUF 格式模型，没法直接加载 PyTorch 或 safetensors 的模型文件。
2. 文档处理只支持 UTF-8 纯文本 TXT，不支持 PDF/EPUB/HTML 等格式。
3. 章节识别正则只覆盖了中文"第X章/Y回/Z卷"和英文"Chapter N"两种模式，其他格式（比如日文"第X話"）可能识别失败。
4. Prompt 超过上下文窗口的时候直接跳过不做回答，不会做截断或压缩处理。
5. 内存占用跟模型大小正相关，没有做量化之外的额外优化。
6. 交互式 `/reload` 默认不切换 Embedding 模型；如果外部修改配置切换了 Embedding 模型，新旧模型维度或语义空间可能不一致，旧数据库需要重新 `--ingest` 建库。
7. 多文档建库时 Chunk ID 是全局递增的，不会区分来自哪本书——metadata 里只存了章节名，如果想区分来自哪本书需要改 DocumentProcessor 在 metadata 里加上书名。
8. llama.cpp 源码依赖 MSVC 私有的 `operator+` 扩展，Linux 编译需要通过 `patches/fs_path_shim.h` 打补丁。这个 bug 的根因在 llama.cpp upstream，补丁只是临时方案。
9. MinGW 对 `std::filesystem` 的链接支持不完整（`codecvt` 等符号缺失），所以 `/reload` 的目录扫描用了平台原生 API 而非标准库。

