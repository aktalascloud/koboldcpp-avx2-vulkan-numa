#include "ggml.h"
#include "gguf.h"
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>

template<class T> void append(std::vector<uint8_t> & b, T v) {
    const auto * p = reinterpret_cast<const uint8_t *>(&v);
    b.insert(b.end(), p, p + sizeof(v));
}
static bool parse_shape(std::array<uint64_t, 4> shape) {
    std::vector<uint8_t> b {'G','G','U','F'};
    append<uint32_t>(b, 3); append<uint64_t>(b, 1); append<uint64_t>(b, 0);
    append<uint64_t>(b, 1); b.push_back('x'); append<uint32_t>(b, 4);
    for (auto n : shape) append(b, n);
    append<uint32_t>(b, GGML_TYPE_F32); append<uint64_t>(b, 0);
    b.resize((b.size() + 31) & ~size_t(31));
    ggml_context * tensors = nullptr;
    auto * g = gguf_init_from_buffer(b.data(), b.size(), {true, &tensors});
    const bool valid = g != nullptr;
    if (g) gguf_free(g);
    if (tensors) ggml_free(tensors);
    return valid;
}
int main() {
    int shapes = 0;
    for (const auto & shape : std::vector<std::array<int64_t,4>> {
            {64,0,1,1}, {0,1,1,1}, {1,0,3,6}, {64,1,0,1},
            {64,2,1,0}, {2,3,1,1}, {1,1,1,1}, {0,0,0,0}}) {
        auto * ctx = ggml_init({1024*1024, nullptr, true}); assert(ctx);
        auto * empty = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, shape[0], shape[1], shape[2], shape[3]);
        ggml_set_name(empty, "shape");
        auto * canary = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        ggml_set_name(canary, "canary");
        std::vector<float> data(ggml_nelements(empty), 2.0f);
        std::array<float,4> marker {1.25f, -9.5f, 123.0f, 0.125f};
        empty->data = data.empty() ? nullptr : data.data(); canary->data = marker.data();
        auto * g = gguf_init_empty(); gguf_add_tensor(g, empty); gguf_add_tensor(g, canary);
        std::vector<uint8_t> bytes(gguf_get_meta_size(g)); gguf_get_meta_data(g, bytes.data());
        const auto start = bytes.size();
        bytes.resize((start + gguf_get_tensor_offset(g, 1) + sizeof(marker) + 31) & ~size_t(31));
        if (!data.empty()) std::memcpy(bytes.data()+start, data.data(), data.size()*sizeof(float));
        std::memcpy(bytes.data()+start+gguf_get_tensor_offset(g, 1), marker.data(), sizeof(marker));
        ggml_context * reloaded = nullptr;
        auto * read = gguf_init_from_buffer(bytes.data(), bytes.size(), {false, &reloaded});
        assert(read && reloaded);
        auto * result = ggml_get_tensor(reloaded, "shape");
        for (int i=0; i<4; ++i) assert(result->ne[i] == shape[i]);
        assert(std::memcmp(ggml_get_tensor(reloaded, "canary")->data, marker.data(), sizeof(marker)) == 0);
        gguf_free(read); ggml_free(reloaded); gguf_free(g); ggml_free(ctx); ++shapes;
    }
    assert(parse_shape({1,0,3,6}));
    assert(!parse_shape({1000, uint64_t(1)<<62, 1, 1}));
    assert(!parse_shape({1000,0,uint64_t(1)<<62,1}));
    assert(!parse_shape({2,uint64_t(INT64_MAX/2),1,1}));
    std::printf("{\"gguf_roundtrips\":%d,\"shape_bounds\":4,\"canary_exact\":true}\n", shapes);
}
