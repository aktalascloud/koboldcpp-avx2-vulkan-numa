#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
//
// hybrid mode (hybrid_threads > 0) keeps the most frequently used experts instead, and does not upload a missing expert
// before it is used: the CPU computes it from host memory while the device computes the cached ones. Missing experts read
// an all-zero slot on the device, so the device's share is exactly zero for them, and the CPU's share is added afterwards.
// On a device behind PCIe 3.0 the upload moves the expert at ~13 GB/s, while the CPU reads it from RAM at ~35 GB/s
// (both measured on the machine this was written for), so computing it in place is the cheaper way to serve a miss.
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
                    int32_t hybrid_threads = 0, int32_t hybrid_uploads = 1);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // hybrid mode: the CPU computes the experts that are not in the cache
    bool hybrid() const;

    // hybrid mode: the CPU's share of the experts of layer il, F32 [n_embd, n_expert_used, n_tokens] in host memory, nullptr if not available
    // the graph adds it to the device's share; the copy callback waits for the CPU before it uploads it
    ggml_tensor * get_cpu_out(int32_t il, int64_t n_tokens) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
