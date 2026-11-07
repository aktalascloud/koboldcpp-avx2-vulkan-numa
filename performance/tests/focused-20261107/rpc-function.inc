ggml_tensor * rpc_server::create_node(uint64_t id,
                                      struct ggml_context * ctx,
                                      const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
    const auto cached = tensor_map.find(id);
    if (cached != tensor_map.end()) {
        return cached->second;
    }
    // First materialize reachable descriptors without C++ recursion. Inserting
    // before visiting edges also handles shared nodes and cycles as before.
    std::vector<uint64_t> pending { id };
    std::vector<uint64_t> created;
    while (!pending.empty()) {
        const uint64_t current = pending.back();
        pending.pop_back();
        if (tensor_map.find(current) != tensor_map.end()) {
            continue;
        }
        const auto descriptor = tensor_ptrs.find(current);
        if (descriptor == tensor_ptrs.end()) {
            return nullptr;
        }
        const rpc_tensor * tensor = descriptor->second;
        ggml_tensor * result = deserialize_tensor(ctx, tensor);
        if (result == nullptr || (result->buffer == nullptr && result->data != nullptr)) {
            LOG_ERROR("[%s] invalid tensor for node id %" PRIu64 "\n", __func__, current);
            return nullptr;
        }
        tensor_map.emplace(current, result);
        created.push_back(current);
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            if (tensor->src[i] != 0) {
                pending.push_back(tensor->src[i]);
            }
        }
        if (tensor->view_src != 0) {
            pending.push_back(tensor->view_src);
        }
    }
    // Then wire edges. Previously memoized tensors are left untouched.
    for (uint64_t current : created) {
        const rpc_tensor * tensor = tensor_ptrs.at(current);
        ggml_tensor * result = tensor_map.at(current);
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            result->src[i] = tensor->src[i] == 0 ? nullptr : tensor_map.at(tensor->src[i]);
        }
        result->view_src = tensor->view_src == 0 ? nullptr : tensor_map.at(tensor->view_src);
        result->view_offs = tensor->view_offs;
    }
    return tensor_map.at(id);
}
