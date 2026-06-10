#include "chunk.h"

#include <cassert>
#include <cstring>

static void writeUint64(FILE *out, uint64_t val) {
    assert(out != nullptr);
    fwrite(&val, sizeof(val), 1, out);
}

static uint64_t readUint64(FILE *in) {
    assert(in != nullptr);
    uint64_t val = 0;
    fread(&val, sizeof(val), 1, in);
    return val;
}

static void writeString(FILE *out, const std::string &s) {
    assert(out != nullptr);
    uint64_t len = static_cast<uint64_t>(s.size());
    fwrite(&len, sizeof(len), 1, out);
    if (len > 0) {
        fwrite(s.data(), 1, len, out);
    }
}

static std::string readString(FILE *in) {
    assert(in != nullptr);
    uint64_t len = readUint64(in);
    std::string s(len, '\0');
    if (len > 0) {
        fread(&s[0], 1, len, in);
    }
    return s;
}

static void writeFloatVec(FILE *out, const std::vector<float> &vec) {
    assert(out != nullptr);
    uint64_t count = static_cast<uint64_t>(vec.size());
    fwrite(&count, sizeof(count), 1, out);
    if (count > 0) {
        fwrite(vec.data(), sizeof(float), count, out);
    }
}

static std::vector<float> readFloatVec(FILE *in) {
    assert(in != nullptr);
    uint64_t count = readUint64(in);
    std::vector<float> vec(count);
    if (count > 0) {
        fread(vec.data(), sizeof(float), count, in);
    }
    return vec;
}

// 序列化格式：id(8B) → text_len+text → meta_len+meta → emb_count+emb_floats
void Chunk::serialize(FILE *out) const {
    assert(out != nullptr);
    assert(!text.empty());
    writeUint64(out, id);
    writeString(out, text);
    writeString(out, metadata);
    writeFloatVec(out, embedding);
}

// 按 serialize 的逆序还原：id ← text ← metadata ← embedding
Chunk Chunk::deserialize(FILE *in) {
    assert(in != nullptr);
    Chunk c;
    c.id = readUint64(in);
    c.text = readString(in);
    c.metadata = readString(in);
    c.embedding = readFloatVec(in);
    return c;
}
