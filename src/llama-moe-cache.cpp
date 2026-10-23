#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif

namespace {

// SLRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
struct moe_cache_slru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // probation and protected LRU lists; hits promote slots to the protected half
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head[2] = { -1, -1 };
    int32_t tail[2] = { -1, -1 };
    std::vector<bool> is_protected;
    int32_t n_protected = 0;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head[0] = 0;
        tail[0] = n_slots - 1;
        is_protected.assign(n_slots, false);
        seen.assign(n_expert, 0);
    }

    // move slot s to the most recently used end of a segment
    void move(int32_t s, bool protect) {
        const bool from = is_protected[s];
        if (from == protect && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[protect];
        next[s] = -1;
        if (tail[protect] >= 0) {
            next[tail[protect]] = s;
        } else {
            head[protect] = s;
        }
        tail[protect] = s;
        is_protected[s] = protect;
        n_protected += int(protect) - int(from);
    }

    void touch(int32_t s) {
        move(s, true);
        if (n_protected > n_slots/2) {
            move(head[1], false);
        }
    }

    int32_t victim(int32_t il) const {
        for (int segment = 0; segment < 2; ++segment) {
            for (int32_t s = head[segment]; s >= 0; s = next[s]) {
                const int32_t key = key_of[s];
                // keep every expert selected by this request, including earlier fills
                if (key < 0 || key / n_expert != il || seen[key % n_expert] != seen_gen) {
                    return s;
                }
            }
        }
        GGML_ABORT("no SLRU victim");
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids in layer il, evicting from probation first
    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit) {
        fills.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // promote hits before choosing victims for misses
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                n_hit++;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            move(s, false);
            fills.push_back({ e, s });
        }
        return true;
    }
};

static inline void moe_cpu_relax() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

// the experts of one layer that are not in the device cache, for every token of a small ubatch
// computed in three stages: the gate and up rows, the activation, the down rows
struct moe_cpu_job {
    struct miss {
        int32_t t;      // token
        int32_t j;      // position among the experts the token selected
        int32_t expert;
    };

    // host weights: gate and up, or gate_up, and down
    const ggml_tensor * gate    = nullptr;
    const ggml_tensor * up      = nullptr;
    const ggml_tensor * gate_up = nullptr;
    const ggml_tensor * down    = nullptr;

    int64_t n_in   = 0; // input width
    int64_t n_ff   = 0;
    int64_t n_out  = 0; // output width
    int64_t n_used = 0;

    std::vector<miss> misses;

    // the layer input of each token, quantized for the vec_dot of the gate (or gate_up) and of the up matrix
    std::vector<uint8_t> x_gate;
    std::vector<uint8_t> x_up;
    size_t x_gate_row = 0;
    size_t x_up_row   = 0;

    std::vector<float>   g; // [n_miss, n_ff] gate, then the activation in place
    std::vector<float>   u; // [n_miss, n_ff] up
    std::vector<uint8_t> h; // [n_miss] activation rows quantized for the vec_dot of down
    size_t h_row = 0;

    float * out = nullptr; // [n_tokens, n_used, n_out]: written at the misses only

    // rows per work item: small enough to balance P- and E-cores, large enough that the counter is not contended
    static constexpr int64_t rows_up   = 16;
    static constexpr int64_t rows_down = 32;

    int64_t n_items[3] = {};
    std::atomic<int64_t> next[3];
    std::atomic<int32_t> arrived[2];
    std::atomic<int32_t> finished;
    std::atomic<bool>    done{true};

    void reset() {
        for (auto & n : next) {
            n.store(0, std::memory_order_relaxed);
        }
        for (auto & a : arrived) {
            a.store(0, std::memory_order_relaxed);
        }
        finished.store(0, std::memory_order_relaxed);
        done.store(false, std::memory_order_relaxed);
    }

    static void barrier(std::atomic<int32_t> & arrived, int32_t n_workers) {
        arrived.fetch_add(1, std::memory_order_acq_rel);
        while (arrived.load(std::memory_order_acquire) < n_workers) {
            moe_cpu_relax();
        }
    }

    static const uint8_t * row(const ggml_tensor * w, int64_t e, int64_t r) {
        return (const uint8_t *) w->data + e*w->nb[2] + r*w->nb[1];
    }

    void run(int32_t n_workers) {
        const int64_t n_miss = (int64_t) misses.size();

        // stage 1: gate and up rows of every miss
        {
            const ggml_tensor * wg = gate_up ? gate_up : gate;
            const ggml_tensor * wu = gate_up ? gate_up : up;
            const auto vd_g = ggml_get_type_traits_cpu(wg->type)->vec_dot;
            const auto vd_u = ggml_get_type_traits_cpu(wu->type)->vec_dot;
            const int64_t per_miss = 2*n_ff;
            const int64_t n_rows   = n_miss*per_miss;
            for (int64_t it; (it = next[0].fetch_add(1, std::memory_order_relaxed)) < n_items[0]; ) {
                const int64_t r1 = std::min(n_rows, (it + 1)*rows_up);
                for (int64_t r = it*rows_up; r < r1; ++r) {
                    const int64_t im = r / per_miss;
                    const int64_t ir = r % per_miss;
                    const miss & m = misses[im];
                    if (ir < n_ff) {
                        vd_g(n_in, &g[im*n_ff + ir], 0, row(wg, m.expert, ir), 0, x_gate.data() + m.t*x_gate_row, 0, 1);
                    } else {
                        // in gate_up the up rows follow the gate rows, in up they start at 0
                        const int64_t ru = gate_up ? ir : ir - n_ff;
                        vd_u(n_in, &u[im*n_ff + ir - n_ff], 0, row(wu, m.expert, ru), 0, x_up.data() + m.t*x_up_row, 0, 1);
                    }
                }
            }
        }
        barrier(arrived[0], n_workers);

        // stage 2: silu(gate)*up, quantized for down
        {
            const ggml_type vdt = ggml_get_type_traits_cpu(down->type)->vec_dot_type;
            const auto from_float = ggml_get_type_traits_cpu(vdt)->from_float;
            for (int64_t im; (im = next[1].fetch_add(1, std::memory_order_relaxed)) < n_items[1]; ) {
                float * a = &g[im*n_ff];
                const float * b = &u[im*n_ff];
                for (int64_t k = 0; k < n_ff; ++k) {
                    a[k] = a[k]/(1.0f + expf(-a[k]))*b[k];
                }
                if (vdt == GGML_TYPE_F32) {
                    memcpy(h.data() + im*h_row, a, n_ff*sizeof(float));
                } else {
                    from_float(a, h.data() + im*h_row, n_ff);
                }
            }
        }
        barrier(arrived[1], n_workers);

        // stage 3: down rows, into the token's slot for that expert
        {
            const auto vd = ggml_get_type_traits_cpu(down->type)->vec_dot;
            const int64_t n_rows = n_miss*n_out;
            for (int64_t it; (it = next[2].fetch_add(1, std::memory_order_relaxed)) < n_items[2]; ) {
                const int64_t r1 = std::min(n_rows, (it + 1)*rows_down);
                for (int64_t r = it*rows_down; r < r1; ++r) {
                    const int64_t im = r / n_out;
                    const int64_t ir = r % n_out;
                    const miss & m = misses[im];
                    vd(n_ff, &out[(m.t*n_used + m.j)*n_out + ir], 0, row(down, m.expert, ir), 0, h.data() + im*h_row, 0, 1);
                }
            }
        }

        if (finished.fetch_add(1, std::memory_order_acq_rel) + 1 == n_workers) {
            done.store(true, std::memory_order_release);
        }
    }
};

// threads that run one moe_cpu_job at a time
// they spin for a while between jobs, because the next layer's job follows within about a millisecond and waking a
// sleeping thread would add tens of microseconds to every layer
class moe_cpu_pool {
public:
    explicit moe_cpu_pool(int32_t n_threads) {
        for (int32_t i = 0; i < n_threads; ++i) {
            workers.emplace_back([this] { work(); });
        }
    }

    ~moe_cpu_pool() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop.store(true);
            gen.fetch_add(1, std::memory_order_release);
        }
        cv.notify_all();
        for (auto & t : workers) {
            t.join();
        }
    }

    void start(moe_cpu_job * j) {
        j->reset();
        {
            std::lock_guard<std::mutex> lock(mtx);
            job = j;
            gen.fetch_add(1, std::memory_order_release);
        }
        cv.notify_all();
    }

    static void wait(const moe_cpu_job & j) {
        while (!j.done.load(std::memory_order_acquire)) {
            moe_cpu_relax();
        }
    }

    int32_t size() const {
        return (int32_t) workers.size();
    }

private:
    static constexpr auto spin_time = std::chrono::milliseconds(5);

    void work() {
        uint64_t seen = 0;
        for (;;) {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; gen.load(std::memory_order_acquire) == seen; ++i) {
                if ((i & 255) == 255 && std::chrono::steady_clock::now() - t0 > spin_time) {
                    std::unique_lock<std::mutex> lock(mtx);
                    cv.wait(lock, [&] { return gen.load(std::memory_order_acquire) != seen; });
                    break;
                }
                moe_cpu_relax();
            }
            moe_cpu_job * j;
            {
                std::lock_guard<std::mutex> lock(mtx);
                seen = gen.load(std::memory_order_acquire);
                j = job;
            }
            if (stop.load()) {
                return;
            }
            j->run((int32_t) workers.size());
        }
    }

    std::vector<std::thread> workers;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<uint64_t> gen{0};
    std::atomic<bool> stop{false};
    moe_cpu_job * job = nullptr;
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_slru slru;

        // hybrid mode
        int32_t null_slot = -1;      // all-zero slot that the experts not in the cache read
        std::vector<float> score;    // [n_layer*n_expert] use count, halved periodically, indexed like the LRU keys
        std::vector<uint8_t> pinned; // [n_slots] uploaded at this step, mapped at the next one
        uint64_t steps = 0;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks

        // hybrid mode
        const ggml_tensor * gate    = nullptr;
        const ggml_tensor * up      = nullptr;
        const ggml_tensor * gate_up = nullptr;
        const ggml_tensor * down    = nullptr;
        ggml_tensor * cpu_out = nullptr;            // F32 [n_out, n_expert_used, max_batch] in host memory
        std::vector<ggml_tensor *> cpu_out_n;       // [max_batch] views of cpu_out for 1..max_batch tokens
        std::vector<std::pair<int32_t, int32_t>> pending; // (expert, slot) uploaded at the last step, mapped at the next
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t bytes  = 0;
    };

    static constexpr int64_t max_batch = 32;

    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    // hybrid mode
    struct hybrid_stats {
        size_t  steps   = 0; // layer steps
        size_t  hits    = 0; // experts the device computed
        size_t  misses  = 0; // experts the CPU computed
        size_t  uploads = 0; // experts admitted into the cache
        size_t  bytes   = 0;
        int64_t t_sync_us = 0; // waiting for the device to finish routing
        int64_t t_wait_us = 0; // waiting for the CPU after the device was given its share
    };
    int32_t hybrid_threads = 0;
    int32_t hybrid_uploads = 0;
    hybrid_stats hst;
    moe_cpu_job job;                       // declared before the pool, so the pool stops first
    std::unique_ptr<moe_cpu_pool> pool;
    bool job_running = false;
    std::vector<float> x_in;               // the layer input read back from the device, if it does not fit in io
    const float * x_cur = nullptr;         // the layer input of the current step, in io or x_in

    // hybrid mode, pinned memory the device reads and writes in its queue: the selected experts and the layer input
    // read back at every layer, and the experts admitted at one step on their way to the banks
    struct pinned_io {
        ggml_backend_buffer_ptr buf;
        int32_t * ids   = nullptr;
        float   * x     = nullptr;
        uint8_t * stage = nullptr;
        size_t n_ids = 0, n_x = 0, stage_size = 0;
        size_t stage_used = 0;  // reset at every step, after the device finished the copies of the previous one
    };
    pinned_io io;
    std::vector<int32_t> candidates;
    std::unordered_map<const ggml_tensor *, int32_t> out_layer; // views of cpu_out -> layer
    ggml_context_ptr ctx_out;              // the views of cpu_out

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_slru::fill> fills;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
         int32_t hybrid_threads, int32_t hybrid_uploads) :
            n_expert_used(model.hparams.n_expert_used_max()),
            hybrid_threads(hybrid_threads), hybrid_uploads(hybrid_uploads),
            layers(model.layers.size()) {
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        if (hybrid_threads > 0 && !hybrid_supported(model)) {
            this->hybrid_threads = 0;
        }
        const int32_t n_null = this->hybrid_threads > 0 ? 1 : 0; // hybrid mode: one all-zero slot after the cached ones

        // one extra slot at the end, CUDA MMQ can read past the last expert
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + n_null + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.slru.init(model.layers.size(), n_expert, g.n_slots);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size()*(n_null + 1);
            if (n_null > 0) {
                g.null_slot = g.n_slots;
                g.score.assign(model.layers.size()*n_expert, 0.0f);
                g.pinned.assign(g.n_slots, 0);
            }
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);
        if (n_null > 0) {
            ctx_out = init_ctx(n_tensors_host*max_batch);
        }

        // hybrid mode copies the slot map and the CPU's share of every layer, from pinned memory the device copies them in
        // its queue; Vulkan copies from pageable memory through a staging buffer and then waits for the device to finish
        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        if (n_null > 0) {
            for (const device & d : devices) {
                ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(d.backend));
                if (buft != nullptr) {
                    buft_host = buft;
                    break;
                }
            }
        }
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + n_null + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots + n_null, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                if (n_null > 0) {
                    const llama_layer & ml = model.layers[il];
                    l.gate    = ml.ffn_gate_exps;
                    l.up      = ml.ffn_up_exps;
                    l.gate_up = ml.ffn_gate_up_exps;
                    l.down    = ml.ffn_down_exps;
                    l.cpu_out = ggml_new_tensor_3d(ctx_host.get(), GGML_TYPE_F32, l.down->ne[1], n_expert_used, max_batch);
                    ggml_format_name(l.cpu_out, "moe_cache.cpu_out-%d", il);
                    buf_host_size += GGML_PAD(ggml_nbytes(l.cpu_out), alignment_host);
                }
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.slru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                        if (g.null_slot >= 0) {
                            std::fill_n(g.slru.slot_map[il], layers[il].slot_map->ne[1], g.null_slot);
                        }
                    }
                }
            }

            // hybrid mode: views of the CPU's share for every ubatch size, so the graph never needs a view of a host weight
            for (layer & l : layers) {
                if (l.cpu_out == nullptr) {
                    continue;
                }
                for (int64_t n = 1; n <= max_batch; ++n) {
                    ggml_tensor * v = ggml_new_tensor_3d(ctx_out.get(), GGML_TYPE_F32, l.cpu_out->ne[0], l.cpu_out->ne[1], n);
                    ggml_format_name(v, "%s.%d", l.cpu_out->name, (int) n);
                    ggml_backend_tensor_alloc(buf_host.get(), v, l.cpu_out->data);
                    l.cpu_out_n.push_back(v);
                    out_layer[v] = (int32_t) (&l - layers.data());
                }
            }
            if (this->hybrid_threads > 0) {
                ggml_cpu_init();
                pool = std::make_unique<moe_cpu_pool>(this->hybrid_threads);

                int64_t n_in = 0;
                size_t expert_bytes = 0;
                for (const layer & l : layers) {
                    if (l.cpu_out == nullptr) {
                        continue;
                    }
                    n_in = std::max(n_in, (l.gate_up ? l.gate_up : l.gate)->ne[0]);
                    size_t sum = 0;
                    for (const ggml_tensor * w : l.experts) {
                        sum += w->nb[2];
                    }
                    expert_bytes = std::max(expert_bytes, sum);
                }
                // admit() uploads at most 4 experts into free slots and hybrid_uploads in place of others per step
                const size_t align = ggml_backend_buft_get_alignment(buft_host);
                io.n_ids      = (size_t) n_expert_used*max_batch;
                io.n_x        = (size_t) n_in*max_batch;
                io.stage_size = (size_t) (4 + std::max(0, this->hybrid_uploads))*expert_bytes;
                const size_t off_x     = GGML_PAD(io.n_ids*sizeof(int32_t), align);
                const size_t off_stage = off_x + GGML_PAD(io.n_x*sizeof(float), align);
                io.buf.reset(ggml_backend_buft_alloc_buffer(buft_host, off_stage + io.stage_size));
                if (io.buf) {
                    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(io.buf.get());
                    io.ids   = (int32_t *) base;
                    io.x     = (float *) (base + off_x);
                    io.stage = base + off_stage;
                } else {
                    io = pinned_io();
                }
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts%s\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0,
                this->hybrid_threads > 0 ? ", hybrid: the CPU computes the missing experts" : "");
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                }
            }
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    ~impl() {
        if (job_running) {
            moe_cpu_pool::wait(job);
        }
        log_stats();
    }

    // hybrid mode computes the experts on the CPU with the vec_dot of each weight type, which needs whole blocks
    static bool hybrid_supported(const llama_model & model) {
        for (const llama_layer & l : model.layers) {
            if (l.ffn_down_exps == nullptr) {
                continue;
            }
            if (l.ffn_gate_exps == nullptr && l.ffn_gate_up_exps == nullptr) {
                LLAMA_LOG_WARN("%s: MoE hybrid mode needs gated experts, disabled\n", __func__);
                return false;
            }
            for (const ggml_tensor * t : { l.ffn_gate_exps, l.ffn_up_exps, l.ffn_gate_up_exps, l.ffn_down_exps }) {
                if (t == nullptr) {
                    continue;
                }
                const auto * tt = ggml_get_type_traits_cpu(t->type);
                if (tt->vec_dot == nullptr || (tt->vec_dot_type != GGML_TYPE_F32 && ggml_get_type_traits_cpu(tt->vec_dot_type)->from_float == nullptr) ||
                    t->ne[0] % ggml_blck_size(tt->vec_dot_type) != 0) {
                    LLAMA_LOG_WARN("%s: MoE hybrid mode cannot compute %s (%s) on the CPU, disabled\n", __func__, t->name, ggml_type_name(t->type));
                    return false;
                }
            }
        }
        return true;
    }

    bool hybrid() const {
        return hybrid_threads > 0 && pool != nullptr;
    }

    ggml_tensor * get_cpu_out(int32_t il, int64_t n_tokens) const {
        if (!hybrid() || il < 0 || il >= (int32_t) layers.size() || n_tokens < 1 || n_tokens > max_batch) {
            return nullptr;
        }
        const layer & l = layers[il];
        return l.ig >= 0 && !l.cpu_out_n.empty() ? l.cpu_out_n[n_tokens - 1] : nullptr;
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.slru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0 || slots[e] == g.null_slot) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        if (hybrid()) {
            const auto it_out = out_layer.find(src);
            if (it_out != out_layer.end()) {
                return copy_cpu_out(backend, src, dst);
            }
        }
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        if (hybrid()) {
            return copy_slots_hybrid(backend, src, dst, graph, il);
        }
        const layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the scheduler calls this before the previous splits are done, and they may still read the banks
        ggml_backend_synchronize(backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        size_t n_hit = 0;
        if (!g.slru.plan(il, ids.data(), ids.size(), fills, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        return true;
    }

    // the get_rows that looks up the slots of the selected experts, and the selected experts
    static const ggml_tensor * find_lookup(const ggml_tensor * dst, ggml_cgraph * graph, int32_t il) {
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));
        return lookup;
    }

    // hybrid mode, the slot map of layer il is about to be copied to the device:
    // read the selected experts and the layer input, give the CPU the experts that are not cached, admit the hottest of
    // them into the cache for the next steps, and copy the slot map, in which the missing experts read the zero slot
    bool copy_slots_hybrid(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, int32_t il) {
        layer & l = layers[il];
        group & g = groups[l.ig];
        GGML_ASSERT(backend == devices[g.id].backend);

        const ggml_tensor * lookup = find_lookup(dst, graph, il);
        const ggml_tensor * sel = lookup->src[1];

        // the first matmul that reads the cache: its second operand is the layer input, one row per token, and its ids
        // give the shape of the selection, which the lookup reads as one flat row
        const ggml_tensor * first = get_experts(l.experts[0]);
        const ggml_tensor * mm = nullptr;
        for (int i = 0; i < ggml_graph_n_nodes(graph) && mm == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_MUL_MAT_ID && node->src[0] == first) {
                mm = node;
            }
        }
        GGML_ASSERT(mm != nullptr);
        const ggml_tensor * xin = mm->src[1];
        const int64_t n_used   = mm->src[2]->ne[0];
        const int64_t n_tokens = mm->src[2]->ne[1];
        GGML_ASSERT(xin->type == GGML_TYPE_F32 && ggml_is_contiguous(xin) && xin->ne[1] == 1 && xin->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(sel) == n_used*n_tokens);

        // the previous layer's job ends in its copy_cpu_out, this only matters if a graph skipped it
        if (job_running) {
            moe_cpu_pool::wait(job);
            job_running = false;
        }

        // into pinned memory both reads are copies in the device's queue and one synchronization waits for them,
        // into pageable memory Vulkan waits for the device at each of them
        const int64_t t0 = ggml_time_us();
        int32_t * p_ids = io.ids;
        float * p_x = io.x;
        if (p_ids == nullptr || (size_t) ggml_nelements(sel) > io.n_ids || (size_t) ggml_nelements(xin) > io.n_x) {
            ids.resize(ggml_nelements(sel));
            x_in.resize(ggml_nelements(xin));
            p_ids = ids.data();
            p_x   = x_in.data();
        }
        ggml_backend_tensor_get_async(backend, sel, p_ids, 0, ggml_nbytes(sel));
        ggml_backend_tensor_get_async(backend, xin, p_x, 0, ggml_nbytes(xin));
        ggml_backend_synchronize(backend);
        hst.t_sync_us += ggml_time_us() - t0;
        x_cur = p_x;
        io.stage_used = 0;

        const int32_t n_expert = (int32_t) l.slot_map->ne[1];
        int32_t * slots = g.slru.slot_map[il];

        // the experts uploaded at the previous step of this layer are in the bank by now: the device ran that upload
        // before anything it was given since, and this step's copies come after it in the same queue
        for (const auto & [e, sl] : l.pending) {
            slots[e] = sl;
            g.pinned[sl] = 0;
        }
        l.pending.clear();

        // hits, misses and use counts
        if (++g.slru.seen_gen == 0) {
            std::fill(g.slru.seen.begin(), g.slru.seen.end(), 0);
            g.slru.seen_gen = 1;
        }
        job.misses.clear();
        for (int64_t t = 0; t < n_tokens; ++t) {
            for (int64_t j = 0; j < n_used; ++j) {
                const int32_t e = p_ids[t*n_used + j];
                GGML_ASSERT(e >= 0 && e < n_expert);
                g.score[(size_t) il*n_expert + e] += 1.0f;
                g.slru.seen[e] = g.slru.seen_gen;
                if (slots[e] != g.null_slot) {
                    hst.hits++;
                } else {
                    job.misses.push_back({ (int32_t) t, (int32_t) j, e });
                }
            }
        }
        hst.misses += job.misses.size();
        hst.steps++;
        // halve the counts every 64 steps of the group's layers, so the cache follows the conversation
        if (++g.steps % (64*g.layers.size()) == 0) {
            for (float & v : g.score) {
                v *= 0.5f;
            }
        }

        // the CPU's share starts at zero, it is only written at the misses
        float * out = (float *) l.cpu_out->data;
        memset(out, 0, (size_t) n_tokens*n_used*l.cpu_out->ne[0]*sizeof(float));
        if (!job.misses.empty()) {
            start_job(l, n_tokens, n_used, out);
        }

        admit(l, g, il, n_expert);

        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));
        return true;
    }

    void start_job(const layer & l, int64_t n_tokens, int64_t n_used, float * out) {
        const ggml_tensor * wg = l.gate_up ? l.gate_up : l.gate;
        const ggml_tensor * wu = l.gate_up ? l.gate_up : l.up;

        job.gate    = l.gate;
        job.up      = l.up;
        job.gate_up = l.gate_up;
        job.down    = l.down;
        job.n_in    = wg->ne[0];
        job.n_ff    = l.down->ne[0];
        job.n_out   = l.down->ne[1];
        job.n_used  = n_used;
        job.out     = out;
        GGML_ASSERT(x_cur != nullptr);

        // quantize the input once per token for each first-stage matrix type
        auto quantize = [&](const ggml_tensor * w, std::vector<uint8_t> & dst, size_t & row) {
            const ggml_type vdt = ggml_get_type_traits_cpu(w->type)->vec_dot_type;
            row = ggml_row_size(vdt, job.n_in);
            dst.resize(row*n_tokens);
            for (int64_t t = 0; t < n_tokens; ++t) {
                if (vdt == GGML_TYPE_F32) {
                    memcpy(dst.data() + t*row, x_cur + t*job.n_in, row);
                } else {
                    ggml_get_type_traits_cpu(vdt)->from_float(x_cur + t*job.n_in, dst.data() + t*row, job.n_in);
                }
            }
        };
        quantize(wg, job.x_gate, job.x_gate_row);
        if (ggml_get_type_traits_cpu(wu->type)->vec_dot_type == ggml_get_type_traits_cpu(wg->type)->vec_dot_type) {
            job.x_up     = job.x_gate;
            job.x_up_row = job.x_gate_row;
        } else {
            quantize(wu, job.x_up, job.x_up_row);
        }

        const int64_t n_miss = (int64_t) job.misses.size();
        job.g.resize(n_miss*job.n_ff);
        job.u.resize(n_miss*job.n_ff);
        job.h_row = ggml_row_size(ggml_get_type_traits_cpu(l.down->type)->vec_dot_type, job.n_ff);
        job.h.resize(n_miss*job.h_row);

        job.n_items[0] = (n_miss*2*job.n_ff + moe_cpu_job::rows_up - 1)/moe_cpu_job::rows_up;
        job.n_items[1] = n_miss;
        job.n_items[2] = (n_miss*job.n_out + moe_cpu_job::rows_down - 1)/moe_cpu_job::rows_down;

        pool->start(&job);
        job_running = true;
    }

    // upload the hottest experts of this step that are not cached, in place of colder ones
    // a free slot is filled up to 4 experts per step; replacing a cached expert is limited to hybrid_uploads per step,
    // and only done if the new expert was used at least once more, so two experts do not keep replacing each other
    void admit(layer & l, group & g, int32_t il, int32_t n_expert) {
        candidates.clear();
        for (const auto & m : job.misses) {
            if (std::find(candidates.begin(), candidates.end(), m.expert) == candidates.end()) {
                candidates.push_back(m.expert);
            }
        }
        const float * score = g.score.data() + (size_t) il*n_expert;
        std::sort(candidates.begin(), candidates.end(), [&](int32_t a, int32_t b) { return score[a] > score[b]; });

        int32_t n_free = 0;
        int32_t n_replaced = 0;
        for (const int32_t e : candidates) {
            int32_t victim = -1;
            float   victim_score = INFINITY;
            for (int32_t sl = 0; sl < g.n_slots; ++sl) {
                if (g.pinned[sl]) {
                    continue;
                }
                const int32_t key = g.slru.key_of[sl];
                if (key < 0) {
                    victim = sl;
                    victim_score = -1.0f;
                    break;
                }
                // an expert this step selected must keep its slot until the device has used it
                if (key / n_expert == il && g.slru.seen[key % n_expert] == g.slru.seen_gen) {
                    continue;
                }
                if (g.score[key] < victim_score) {
                    victim_score = g.score[key];
                    victim = sl;
                }
            }
            if (victim < 0) {
                break;
            }
            const bool is_free = victim_score < 0.0f;
            if (is_free ? n_free >= 4 : (n_replaced >= hybrid_uploads || score[e] < victim_score + 1.0f)) {
                break; // the remaining candidates are colder
            }
            (is_free ? n_free : n_replaced)++;

            const int32_t old = g.slru.key_of[victim];
            if (old >= 0) {
                g.slru.slot_map[old / n_expert][old % n_expert] = g.null_slot;
            }
            g.slru.key_of[victim] = il*n_expert + e;
            g.pinned[victim] = 1;
            l.pending.emplace_back(e, victim);

            ggml_backend_t backend = devices[g.id].backend;
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w = l.experts[ip];
                const size_t expert_size = w->nb[2];
                const uint8_t * data = (const uint8_t *) w->data + e*expert_size;
                // staged in pinned memory, the copy waits in the device's queue instead of the host waiting for the device
                if (io.stage != nullptr && io.stage_used + expert_size <= io.stage_size) {
                    uint8_t * p = io.stage + io.stage_used;
                    memcpy(p, data, expert_size);
                    io.stage_used += expert_size;
                    data = p;
                }
                ggml_backend_tensor_set_async(backend, g.banks[ip], data, victim*expert_size, expert_size);
                hst.bytes += expert_size;
            }
            hst.uploads++;
        }
    }

    // hybrid mode, the CPU's share of a layer is about to be copied to the device: wait for the CPU to finish it
    bool copy_cpu_out(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst) {
        if (job_running) {
            const int64_t t0 = ggml_time_us();
            moe_cpu_pool::wait(job);
            hst.t_wait_us += ggml_time_us() - t0;
            job_running = false;
        }
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));
        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (hst.steps > 0) {
            const size_t n = hst.hits + hst.misses;
            LLAMA_LOG_INFO("llama_moe_cache: hybrid: %zu layer steps, device %zu, CPU %zu experts (device share %.2f%%), "
                "%zu uploads (%.2f MiB), routing sync %.2f ms/step, CPU wait %.2f ms/step\n",
                hst.steps, hst.hits, hst.misses, n ? 100.0*hst.hits/n : 0.0, hst.uploads, hst.bytes/1024.0/1024.0,
                hst.t_sync_us/1000.0/hst.steps, hst.t_wait_us/1000.0/hst.steps);
        }
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
                                 int32_t hybrid_threads, int32_t hybrid_uploads) :
    pimpl(new impl(model, backends, bufts, size, hybrid_threads, hybrid_uploads)) {
}

bool llama_moe_cache::hybrid() const {
    return pimpl->hybrid();
}

ggml_tensor * llama_moe_cache::get_cpu_out(int32_t il, int64_t n_tokens) const {
    return pimpl->get_cpu_out(il, n_tokens);
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
