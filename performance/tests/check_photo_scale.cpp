// Manual GPU/CPU check: a row view must equal materialize-then-SCALE bit for bit.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check(ggml_backend_t backend, int width, int rows, int padding, int offset,
                  float scale, float bias) {
    const int pitch = 2 * width + padding;
    auto ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    if (!ctx) throw std::runtime_error("context allocation failed");
    auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pitch, rows);
    auto view = ggml_view_2d(ctx, input, width, rows, pitch * sizeof(float), offset * sizeof(float));
    auto reference = ggml_scale_bias(ctx, ggml_cont(ctx, view), scale, bias);
    auto result = ggml_scale_bias(ctx, view, scale, bias);
    ggml_set_input(input);
    ggml_set_output(reference);
    ggml_set_output(result);
    if (!ggml_backend_supports_op(backend, result)) throw std::runtime_error("SCALE view unsupported");
    auto graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, reference);
    ggml_build_forward_expand(graph, result);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) throw std::runtime_error("backend allocation failed");
    std::vector<float> data(size_t(pitch) * rows);
    for (size_t i = 0; i < data.size(); ++i) data[i] = float(int((i * 7919 + 13) % 65521) - 32760) / 997.f;
    ggml_backend_tensor_set(input, data.data(), 0, data.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("compute failed");
    std::vector<float> expected(size_t(width) * rows), actual(expected.size());
    ggml_backend_tensor_get(reference, expected.data(), 0, expected.size() * sizeof(float));
    ggml_backend_tensor_get(result, actual.data(), 0, actual.size() * sizeof(float));
    if (memcmp(expected.data(), actual.data(), expected.size() * sizeof(float))) {
        std::fprintf(stderr, "SCALE mismatch %s width=%d rows=%d pad=%d offset=%d scale=%g bias=%g\n",
                     ggml_backend_name(backend), width, rows, padding, offset, scale, bias);
        throw std::runtime_error("non-bit-exact SCALE result");
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main() {
    try {
        auto cpu = ggml_backend_cpu_init();
        auto gpu = ggml_backend_vk_init(0);
        if (!cpu || !gpu) throw std::runtime_error("both CPU and Vulkan required");
        ggml_backend_cpu_set_n_threads(cpu, 4);
        int cases = 0;
        for (auto backend : {cpu, gpu}) {
            for (int width : {1, 3, 32, 127, 8192}) {
                for (int rows : {1, 7, 97}) {
                    for (float scale : {0.25f, 1.f / 25.f, -2.f}) {
                        for (float bias : {0.f, 0.125f}) {
                            check(backend, width, rows, 3, width, scale, bias);
                            ++cases;
                        }
                    }
                }
            }
            check(backend, 8192, 3236, 0, 0, 0.25f, 0.f); ++cases;
            check(backend, 8192, 5108, 0, 8192, 1.f / 25.f, 0.f); ++cases;
        }
        ggml_backend_free(gpu); ggml_backend_free(cpu);
        std::printf("{\"ok\":true,\"bit_exact_cases\":%d,\"backends\":[\"CPU\",\"Vulkan\"]}\n", cases);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
