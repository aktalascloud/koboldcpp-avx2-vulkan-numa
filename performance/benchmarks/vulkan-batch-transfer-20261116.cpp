#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    const auto device = ggml_backend_vk_init(0);
    if (!device) return 1;

    for (const size_t bytes : {4096, 65536, 1048576}) {
        constexpr size_t count = 24;
        auto * ctx = ggml_init({1024*1024, nullptr, true});
        if (!ctx) return 2;
        std::vector<ggml_tensor *> tensors(count);
        std::vector<size_t> offsets(count, 16), sizes(count, bytes), host_offsets(count);
        std::vector<uint8_t> input(count*bytes), output(count*bytes);
        for (size_t i = 0; i < count; ++i) {
            tensors[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, bytes+32);
            host_offsets[i] = i*bytes;
        }
        std::vector<const ggml_tensor *> read_tensors(tensors.begin(), tensors.end());
        const auto buffer = ggml_backend_alloc_ctx_tensors(ctx, device);
        if (!buffer) return 3;

        auto verify = [&](bool batch) {
            if (batch) {
                ggml_backend_tensor_get_n(read_tensors.data(), output.data(),
                    offsets.data(), sizes.data(), host_offsets.data(), count);
            } else {
                for (size_t i = 0; i < count; ++i) {
                    ggml_backend_tensor_get(tensors[i], output.data()+host_offsets[i], offsets[i], bytes);
                }
            }
        };
        auto upload = [&](bool batch) {
            if (batch) {
                ggml_backend_tensor_set_n(tensors.data(), input.data(),
                    offsets.data(), sizes.data(), host_offsets.data(), count);
            } else {
                for (size_t i = 0; i < count; ++i) {
                    ggml_backend_tensor_set(tensors[i], input.data()+host_offsets[i], offsets[i], bytes);
                }
            }
        };
        uint32_t state = 0x518def33;
        for (auto & b : input) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            b = uint8_t(state);
        }

        for (const bool batch : {false, true}) {
            for (auto & b : input) b ^= 0xff;
            std::fill(output.begin(), output.end(), 0);
            upload(batch);
            verify(!batch);
            if (input != output) return 4;
        }

        for (const bool read : {false, true}) {
            for (int i = -2; i < 9; ++i) {
                for (const int order : {0, 1}) {
                    const bool batch = (i & 1) ^ order;
                    const auto start = std::chrono::steady_clock::now();
                    for (int repeat = 0; repeat < 5; ++repeat) {
                        if (read) verify(batch);
                        else upload(batch);
                    }
                    const auto end = std::chrono::steady_clock::now();
                    if (i >= 0) {
                        std::printf("sample %s %zu %zu %d %d %.3f\n", read ? "get" : "set", count,
                            bytes, i, batch, std::chrono::duration<double,std::micro>(end-start).count()/5.0);
                    }
                }
            }
        }
        verify(true);
        if (input != output) return 5;
        std::printf("exact %zu %zu\n", count, bytes);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }

    ggml_backend_free(device);
}
