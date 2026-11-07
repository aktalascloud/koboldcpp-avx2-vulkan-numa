#if defined(__x86_64__) || defined(_M_X64)
template<typename GetLogit>
static size_t llama_sampler_greedy_select(size_t size, GetLogit get_logit) {
    if (size <= 1) {
        return 0;
    }
    // Start every accumulator at the first candidate, preserving the original
    // first-NaN and lowest-index tie rules.
    constexpr size_t width = 4;
    std::array<float, width> maxima;
    maxima.fill(get_logit(0));
    std::array<size_t, width> indices = {};
    size_t i = 1;
    for (; size - i >= width; i += width) {
        for (size_t lane = 0; lane < width; ++lane) {
            const float logit = get_logit(i + lane);
            if (logit > maxima[lane]) {
                maxima[lane] = logit;
                indices[lane] = i + lane;
            }
        }
    }
    size_t selected = indices[0];
    float maximum = maxima[0];
    for (size_t lane = 1; lane < width; ++lane) {
        if (maxima[lane] > maximum || (maxima[lane] == maximum && indices[lane] < selected)) {
            maximum = maxima[lane];
            selected = indices[lane];
        }
    }
    for (; i < size; ++i) {
        if (get_logit(i) > maximum) {
            maximum = get_logit(i);
            selected = i;
        }
    }
    return selected;
}

static void llama_sampler_greedy_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p);
#endif

static void apply_before(struct llama_sampler * /*smpl*/, llama_token_data_array * cur_p) {
    cur_p->selected = 0;
    for (size_t i = 1; i < cur_p->size; ++i) {
        if (cur_p->data[i].logit > cur_p->data[cur_p->selected].logit) {
            cur_p->selected = i;
        }
    }
}
static void llama_sampler_greedy_apply(struct llama_sampler * /*smpl*/, llama_token_data_array * cur_p) {
#if defined(__x86_64__) || defined(_M_X64)
    cur_p->selected = llama_sampler_greedy_select(cur_p->size, [&](size_t i) { return cur_p->data[i].logit; });
#else
    cur_p->selected = 0;
    for (size_t i = 1; i < cur_p->size; ++i) {
        if (cur_p->data[i].logit > cur_p->data[cur_p->selected].logit) {
            cur_p->selected = i;
        }
    }
#endif
}
llama_token sample_before(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx) {
    const llama_token   sampled_token  = llama_get_sampled_token_ith     (ctx, idx);
    const float *       sampled_probs  = llama_get_sampled_probs_ith     (ctx, idx);
    const float *       sampled_logits = llama_get_sampled_logits_ith    (ctx, idx);
    const llama_token * sampled_ids    = llama_get_sampled_candidates_ith(ctx, idx);

    // If a backend sampler has already sampled a token, return it.
    if (sampled_token != LLAMA_TOKEN_NULL) {
        LLAMA_LOG_DEBUG("%s: Backend sampler selected token for idx %d. Skipping CPU samplers\n", __func__, idx);
        llama_sampler_accept(smpl, sampled_token);
        return sampled_token;
    }

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_vocab = llama_vocab_n_tokens(vocab);

    // use pre-allocated buffer from chain if available, otherwise allocate locally
    std::vector<llama_token_data> * cur_ptr;
    std::vector<llama_token_data> cur_local;

    if (smpl->iface == &llama_sampler_chain_i) {
        auto * chain = (llama_sampler_chain *) smpl->ctx;
        cur_ptr = &chain->cur;
    } else {
        cur_ptr = &cur_local;
    }

    auto & cur = *cur_ptr;

    if (sampled_probs) {
        const uint32_t sampled_probs_count = llama_get_sampled_probs_count_ith(ctx, idx);
        cur.resize(sampled_probs_count);
        for (uint32_t i = 0; i < sampled_probs_count; ++i) {
            cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], sampled_probs[i]};
        }
    } else if (sampled_logits) {
        const uint32_t sampled_logits_count = llama_get_sampled_logits_count_ith(ctx, idx);
        cur.resize(sampled_logits_count);
        for (llama_token i = 0; i < (int)sampled_logits_count; i++) {
            cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], 0.0f};
        }
    } else {
        const auto * logits = llama_get_logits_ith(ctx, idx);
        GGML_ASSERT(logits != nullptr);
        cur.resize(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
        }
    }

    llama_token_data_array cur_p = {
        /* .data       = */ cur.data(),
        /* .size       = */ cur.size(),
        /* .selected   = */ -1,
        /* .sorted     = */ false,
    };

    llama_sampler_apply(smpl, &cur_p);

    GGML_ASSERT(cur_p.selected >= 0 && cur_p.selected < (int32_t) cur_p.size);

    auto token = cur_p.data[cur_p.selected].id;

    llama_sampler_accept(smpl, token);

    return token;
}

llama_token sample_after(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx) {
    const llama_token   sampled_token  = llama_get_sampled_token_ith     (ctx, idx);
    const float *       sampled_probs  = llama_get_sampled_probs_ith     (ctx, idx);
    const float *       sampled_logits = llama_get_sampled_logits_ith    (ctx, idx);
    const llama_token * sampled_ids    = llama_get_sampled_candidates_ith(ctx, idx);

    // If a backend sampler has already sampled a token, return it.
    if (sampled_token != LLAMA_TOKEN_NULL) {
        LLAMA_LOG_DEBUG("%s: Backend sampler selected token for idx %d. Skipping CPU samplers\n", __func__, idx);
        llama_sampler_accept(smpl, sampled_token);
        return sampled_token;
    }

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_vocab = llama_vocab_n_tokens(vocab);

#if defined(__x86_64__) || defined(_M_X64)
    // A bare greedy sampler consumes only logits. Keep compact backend IDs and
    // the single accept callback while avoiding a vocabulary-sized candidate copy.
    if (smpl->iface->apply == llama_sampler_greedy_apply) {
        const float * logits = sampled_logits ? sampled_logits : llama_get_logits_ith(ctx, idx);
        const size_t count = sampled_probs ? llama_get_sampled_probs_count_ith(ctx, idx) :
                             sampled_logits ? llama_get_sampled_logits_count_ith(ctx, idx) : n_vocab;
        GGML_ASSERT(logits != nullptr && count > 0);
        GGML_ASSERT(!sampled_probs || sampled_logits);
        GGML_ASSERT(!sampled_logits || sampled_ids);
        const size_t selected = llama_sampler_greedy_select(count, [&](size_t i) { return logits[i]; });
        const llama_token token = sampled_logits ? sampled_ids[selected] : static_cast<llama_token>(selected);
        llama_sampler_accept(smpl, token);
        return token;
    }
#endif

    // use pre-allocated buffer from chain if available, otherwise allocate locally
    std::vector<llama_token_data> * cur_ptr;
    std::vector<llama_token_data> cur_local;

    if (smpl->iface == &llama_sampler_chain_i) {
        auto * chain = (llama_sampler_chain *) smpl->ctx;
        cur_ptr = &chain->cur;
    } else {
        cur_ptr = &cur_local;
    }

    auto & cur = *cur_ptr;

    if (sampled_probs) {
        const uint32_t sampled_probs_count = llama_get_sampled_probs_count_ith(ctx, idx);
        cur.resize(sampled_probs_count);
        for (uint32_t i = 0; i < sampled_probs_count; ++i) {
            cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], sampled_probs[i]};
        }
    } else if (sampled_logits) {
        const uint32_t sampled_logits_count = llama_get_sampled_logits_count_ith(ctx, idx);
        cur.resize(sampled_logits_count);
        for (llama_token i = 0; i < (int)sampled_logits_count; i++) {
            cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], 0.0f};
        }
    } else {
        const auto * logits = llama_get_logits_ith(ctx, idx);
        GGML_ASSERT(logits != nullptr);
        cur.resize(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
        }
    }

    llama_token_data_array cur_p = {
        /* .data       = */ cur.data(),
        /* .size       = */ cur.size(),
        /* .selected   = */ -1,
        /* .sorted     = */ false,
    };

    llama_sampler_apply(smpl, &cur_p);

    GGML_ASSERT(cur_p.selected >= 0 && cur_p.selected < (int32_t) cur_p.size);

    auto token = cur_p.data[cur_p.selected].id;

    llama_sampler_accept(smpl, token);

    return token;
}
