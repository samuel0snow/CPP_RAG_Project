# MRAG 大作业验收复习

这份文档按老师给的验收流程整理，面试前可以快速过一遍：先记住怎么演示，再记住每个模块为什么这样设计，最后准备几个高频追问。

## 一、验收前准备清单

### 1. 必须确认的文件

- 源码：`src/`
- 构建脚本：`CMakeLists.txt`
- 配置文件：`config.json`
- 数据文件：`data/*.txt`
- 已建好的向量数据库：`data/sanguo.db`、`data/honglou.db`、`data/shuihu.db`、`data/xiyou.db`
- 第三方库：`resources/llama.cpp-master/`
- 模型目录：`models/`
- Valgrind 截图或报告：`Valgrind检测.pdf`

当前项目里已经有建好的数据库、Valgrind PDF 和 `models/` 目录。验收机器上至少需要两个默认模型：

```text
models/bge-small-zh-v1.5-f16.gguf
models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

如果模型文件缺失，项目可以编译，但 `--ingest` 和 `--chat` 都会在加载模型时失败。

当前已额外下载 7B 模型，用来演示 `/reload` 热切换：

```text
models/qwen2.5-7b-instruct-q4_k_m.gguf
```

如果需要重新下载资源，推荐国内网络使用镜像：

```bash
python download.py --skip-novels --skip-llama --hf-endpoint https://hf-mirror.com
```

### 2. 推荐现场命令

在项目根目录执行：

```bash
cmake -S . -B build
cmake --build build
```

Windows 下运行：

```bash
.\build\mrag.exe --chat .\data\sanguo.db
```

Linux 下运行：

```bash
./build/mrag --chat ./data/sanguo.db
```

如果老师要求看建库流程，可以用较小的文本提前或现场演示：

```bash
.\build\mrag.exe --ingest .\data\三国演义.txt .\data\sanguo_demo.db
```

多文档建库演示：

```bash
.\build\mrag.exe --ingest .\data\三国演义.txt .\data\水浒传.txt .\data\西游记.txt .\data\红楼梦.txt .\data\four_classics.db
```

命令规则：`--ingest` 后最后一个参数是输出 `.db`，前面的都是输入 `.txt`。

## 二、按验收流程怎么讲

### 1. 建库可以提前完成

老师允许不同同学文本长度不同，建库时间不一致，所以数据库可以提前生成。我们的建库产物在 `data/` 目录，重点展示 `.db` 文件存在，并说明它是二进制向量数据库。

可讲：

> 我们把小说 TXT 先切成 Chunk，再用 Embedding 模型生成向量，最后把 Chunk 文本、章节 metadata 和 embedding 一起序列化到 `.db` 文件里。验收时为了节约时间，可以直接加载已建好的 `data/sanguo.db`。

### 2. 检查 `--ingest`

入口在 `src/mragapp.cc` 的 `buildKnowledgeBase()`。

完整流程：

1. `loadConfig("config.json")` 读取配置。
2. `DocumentProcessor::processNovel()` 逐本文档切分。
3. 加载 `EmbeddingEngine`。
4. 对每个 Chunk 调 `generateEmbedding()`。
5. `VectorDatabase::insert()` 存入内存。
6. `VectorDatabase::saveToDisk()` 写入 `.db`。

运行时会看到类似输出：

```text
[INFO] 正在处理文档: ...
[INFO] 当前文档切分完成，共 ... 个片段
[INFO] 正在生成向量嵌入...
[INFO] 知识库构建完成，保存至: ...
[INFO] 数据库包含 ... 个向量条目
```

### 3. 检查 `--chat`

入口在 `src/mragapp.cc` 的 `chatLoop()`。

完整流程：

1. 从 `.db` 加载向量数据库。
2. 加载 Embedding 模型。
3. 加载生成模型。
4. 用户输入问题。
5. 将问题转成 query embedding。
6. `VectorDatabase::search()` 检索 top-k 片段。
7. 打印召回片段的章节来源和文本预览。
8. `GenerationEngine::generateStream()` 构造 prompt 并调用 llama.cpp 生成回答。

问答时会先打印：

```text
[检索到 3 个相关片段]
  [1] 第一回 ...: 片段预览...
  [2] ...
  [3] ...
```

这正好对应老师强调的“为了检查向量检索结果，请在问答时打印召回文本块内容或来源”。

### 4. 两轮问答推荐问题

推荐先问短事实题，容易看出检索来源是否合理：

```text
刘备的字是什么？
关羽的武器是什么？
桃园结义发生在哪里？
十常侍是哪些人？
```

如果使用 `data/sanguo.db`，这些问题通常能检索到前几回相关片段。回答时不要只看大模型答案，也要主动指出“上面打印的片段来源说明检索命中了相关章节”。

### 5. 加分项怎么演示

配置文件热切换：

- 在 `--chat` 中输入 `/reload`。
- 程序会扫描 `models/` 目录下的 `.gguf` 文件。
- 先显示生成模型菜单，再显示 Embedding 模型菜单。
- 如果切换 Embedding 模型，会重新计算当前数据库中全部 Chunk 的向量。

多文档建库：

- `--ingest` 支持多个 TXT。
- 最后一个参数是输出 DB，前面所有参数都是输入 TXT。

Valgrind：

- 提交目录里有 `Valgrind检测.pdf`。
- 可以说主要检查了程序运行过程是否有明显内存泄漏。

## 三、核心模块复习

### 1. `main.cc`

只做一件事：

```cpp
return runMragApp(argc, argv);
```

这样把标准入口和业务逻辑分开，便于维护。

### 2. `mragapp.cc`

应用层总控，负责：

- 命令行解析：`--ingest`、`--chat`、`--query`
- 初始化 llama backend
- 建库流程
- 交互问答循环
- `/reload` 模型热切换
- 查询时打印检索片段

可以把它理解成“把所有模块串起来的流程层”。

### 3. `document.cc`

负责把小说 TXT 切成 Chunk：

- 按行读取文本。
- 用章节标题规则识别“第 X 回 / 第 X 章 / Chapter N”。
- 按 `chunk_size` 切块。
- 使用 `overlap_size` 做重叠，避免答案刚好落在切割边界。
- 用 UTF-8 边界判断避免中文被切坏。

老师如果问为什么要 overlap：

> 因为 RAG 检索的最小单位是 Chunk。如果一句关键证据刚好被切成两半，单个片段就可能不完整。重叠可以让边界附近的内容同时出现在相邻片段里，提高召回稳定性。

### 4. `embeddingengine.cc`

负责“文本转向量”：

- 清理 UTF-8。
- tokenize。
- 截断到 embedding 模型上下文长度。
- 清空 KV cache。
- 调 llama.cpp 推理。
- 取出 embedding 向量。

建库时对每个 Chunk 做一次 embedding；问答时对用户问题做一次 embedding。

### 5. `vectordatabase.cc`

负责保存、加载、检索：

- 数据结构是 `std::vector<Chunk>`。
- `.db` 格式包含 magic、version、count 和多个 Chunk。
- 检索使用余弦相似度 + 关键词分数 + 一些事实题加权。
- 用 `priority_queue` 维护 top-k，复杂度是 `O(n log k)`。

老师如果问为什么不用全排序：

> 因为每次只需要 top-k，比如 k=3。用最小堆可以边遍历边维护最优的 k 个结果，比对全部 Chunk 排序更省。

### 6. `generationengine.cc`

负责根据检索片段生成回答：

- 把检索到的 Chunk 组织成带来源的上下文。
- 用 ChatML 构造 system/user/assistant prompt。
- system prompt 要求模型只能根据上下文回答，证据不足就说无法得到答案。
- prefill 整个 prompt。
- 采样链顺序是 top-k、top-p、temperature、dist。
- 循环生成 token，直到结束符或达到最大输出长度。

老师如果问为什么 prompt 要强调不能猜：

> RAG 的答案质量取决于召回片段。如果片段没有证据，大模型可能凭常识或训练记忆乱答。所以 prompt 中要求它只依据当前上下文，证据不足时保守回答。

### 7. `config.cc`

负责读取和保存 `config.json`：

- 模型路径
- `chunk_size`、`overlap_size`
- `retrieval.top_k`
- 生成参数
- embedding 参数

热切换时，程序会重新读取并写回 `config.json`。

### 8. `llamamodel.cc`

封装 llama.cpp 的公共部分：

- 加载 GGUF 模型。
- 创建 llama context。
- 保存 vocab。
- 封装 tokenize。
- 用 RAII 在析构时释放模型和上下文。

老师如果问 RAII 的意义：

> 模型和上下文是 C API 资源，需要手动释放。RAII 可以保证对象生命周期结束时自动释放，减少泄漏和重复释放风险。

## 四、老师可能追问

### Q1：这个项目的完整 RAG 流程是什么？

A：离线阶段先把 TXT 切成 Chunk，用 Embedding 模型把每个 Chunk 转成向量，然后保存为 `.db`。在线阶段加载 `.db`，用户问题也转成向量，向量数据库检索 top-k 相关 Chunk，把这些 Chunk 和问题拼进 prompt，最后由本地生成模型输出答案。

### Q2：向量数据库里存了什么？

A：每条记录是一个 Chunk，包含 `id`、`text`、`metadata` 和 `embedding`。`text` 是片段原文，`metadata` 主要是章节来源，`embedding` 是浮点向量。

### Q3：为什么检索时还要加关键词分数？

A：纯向量相似度有时对短事实问题不稳定，比如问“刘备的字是什么”，正确片段必须包含“刘备/玄德/字”这类直接证据。关键词和事实题加权可以把包含明确证据的片段排得更靠前。

### Q4：Embedding 模型切换后为什么要重建向量？

A：不同 Embedding 模型生成的向量不在同一个语义空间里。即使维度一样，也不能拿新模型的问题向量去查旧模型生成的数据库向量。所以切换 Embedding 后必须重新计算当前数据库所有 Chunk 的 embedding。

### Q5：为什么默认用 1.5B，不默认用 7B？

A：7B 模型更大，加载和推理都更吃内存和显存。验收首先要保证稳定运行，所以默认用 Qwen2.5-1.5B；7B 作为可选模型，可以在机器资源足够时通过 `/reload` 演示。

### Q6：多文档建库怎么实现？

A：`--ingest` 中最后一个参数固定作为输出 DB，前面的所有参数都当作输入 TXT。程序逐个处理每本书，把所有 Chunk 合并到同一个 `VectorDatabase` 后统一写盘。

### Q7：检索结果怎么给老师看？

A：每次问答时，`doQuery()` 会打印 `[检索到 N 个相关片段]`，然后列出每个片段的 `metadata` 和前 80 字预览。生成 prompt 里也会使用 `[片段 i，出自：metadata]` 的形式带上来源。

### Q8：如果模型回答错了怎么办？

A：先看检索片段是否命中。如果检索片段不包含答案，说明召回失败，需要调整切块、top-k 或检索评分。如果检索片段包含答案但模型没答对，说明生成阶段不稳定，可以加强 prompt、换模型或降低温度。

### Q9：为什么 Chunk 要序列化成二进制？

A：embedding 是大量 float，二进制保存比文本 JSON 更紧凑，加载也更快。文件头用 magic 和 version 校验，可以避免误读错误格式的文件。

### Q10：Linux 编译补丁是干什么的？

A：llama.cpp 中有些地方用了 `std::filesystem::path + "string"`，这是 MSVC 能接受但 GCC 不接受的非标准写法。项目用 `patches/fs_path_shim.h` 补了兼容 operator，并在非 MSVC 编译时自动 include。

## 五、当前项目检查结论

### 已确认通过

- `cmake -S . -B build` 配置通过。
- `cmake --build build` 编译通过，生成 `build/mrag.exe`。
- 无参数运行能正常打印 `--ingest`、`--chat`、`--query` 用法。
- `data/` 中已有多个 `.db` 文件，便于跳过长时间建库直接验收问答。
- 问答流程中已经打印检索片段来源和预览，符合老师提醒。
- 项目支持 `/reload`、多文档建库、Valgrind PDF 三个加分展示点。

### 现场风险

- `config.json` 默认 `n_gpu_layers` 是 99。如果验收机器没有合适 GPU 或 llama.cpp 后端回退不稳定，可以改成 0 使用 CPU：

```json
"n_gpu_layers": 0
```

- 多文档大库建库时间较长，建议提前准备 `.db`，现场只做小规模演示。

## 六、最后一分钟背诵版

项目是本地 RAG 小说问答系统。`--ingest` 做离线建库：TXT 读取、章节识别、UTF-8 安全切块、重叠窗口、Embedding 生成、二进制 DB 保存。`--chat` 做在线问答：加载 DB 和模型、问题转向量、向量检索 top-k、打印召回片段来源、把片段和问题构造成 prompt、由本地 GGUF 生成模型回答。检索不是只靠余弦相似度，还加了关键词和事实题证据加权。`/reload` 支持运行时切换模型，切 Embedding 时会重建当前数据库向量。默认用 1.5B 是为了稳定验收，7B 是资源允许时的加分演示。
