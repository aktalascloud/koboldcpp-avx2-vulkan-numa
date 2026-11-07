#include "ggml-impl.h"
#include "otherarch/sdcpp/src/core/ggml_graph_cut.h"
namespace sd::ggml_graph_cut {
    int leaf_count(ggml_cgraph* gf) {
        GGML_ASSERT(gf != nullptr);
        return gf->n_leafs;
    }
    ggml_tensor* leaf_tensor(ggml_cgraph* gf, int leaf_index) {
        GGML_ASSERT(gf != nullptr);
        if (leaf_index < 0 || leaf_index >= gf->n_leafs) {
            return nullptr;
        }
        return gf->leafs[leaf_index];
    }
    ggml_tensor* input_tensor(ggml_cgraph* gf, const Segment::InputRef& input_ref) {
        GGML_ASSERT(gf != nullptr);
        if (input_ref.type == Segment::INPUT_PREVIOUS_CUT) {
            if (input_ref.node_index < 0 || input_ref.node_index >= ggml_graph_n_nodes(gf)) {
                return nullptr;
            }
            return ggml_graph_node(gf, input_ref.node_index);
        }
        if (input_ref.leaf_index < 0 || input_ref.leaf_index >= gf->n_leafs) {
            return nullptr;
        }
        return leaf_tensor(gf, input_ref.leaf_index);
    }
    ggml_cgraph* build_segment_graph(ggml_cgraph* gf,
                                     const Segment& segment,
                                     ggml_context** graph_ctx_out) {
        GGML_ASSERT(gf != nullptr);
        GGML_ASSERT(graph_ctx_out != nullptr);

        // Collect leaf inputs and internal nodes, then any tensor they
        // reference that is not already represented, notably the view_src of a
        // view-typed input leaf. ggml_gallocr sizes its hash set from
        // n_nodes + n_leafs (plus a 25% margin that rounds down to zero for a
        // one-node segment), so every distinct tensor it will hash must be
        // counted here or a tiny segment overflows the hash set and aborts.
        std::vector<ggml_tensor*> leaves;
        std::unordered_set<ggml_tensor*> represented;
        for (const auto& input : segment.input_refs) {
            ggml_tensor* current_input = input_tensor(gf, input);
            if (current_input == nullptr) {
                continue;
            }
            if (represented.insert(current_input).second) {
                leaves.push_back(current_input);
            }
        }
        for (int node_idx : segment.internal_node_indices) {
            represented.insert(ggml_graph_node(gf, node_idx));
        }
        auto add_reference = [&](ggml_tensor* tensor) {
            if (tensor != nullptr && represented.insert(tensor).second) {
                leaves.push_back(tensor);
            }
        };
        for (int node_idx : segment.internal_node_indices) {
            ggml_tensor* node = ggml_graph_node(gf, node_idx);
            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                add_reference(node->src[src_idx]);
            }
            add_reference(node->view_src);
        }
        for (size_t i = 0; i < leaves.size(); ++i) {
            add_reference(leaves[i]->view_src);
        }

        const size_t graph_size = segment.internal_node_indices.size() + leaves.size() + 8;
        ggml_init_params params = {
            /*.mem_size   =*/ggml_graph_overhead_custom(graph_size, false) + 1024,
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };
        ggml_context* graph_ctx = ggml_init(params);
        GGML_ASSERT(graph_ctx != nullptr);
        ggml_cgraph* segment_graph = ggml_new_graph_custom(graph_ctx, graph_size, false);
        GGML_ASSERT(segment_graph != nullptr);

        for (ggml_tensor* leaf : leaves) {
            GGML_ASSERT(segment_graph->n_leafs < segment_graph->size);
            segment_graph->leafs[segment_graph->n_leafs++] = leaf;
        }

        for (int output_node_index : segment.output_node_indices) {
            ggml_tensor* output = ggml_graph_node(gf, output_node_index);
            if (output == nullptr) {
                continue;
            }
            ggml_set_output(output);
            if (output->view_src != nullptr) {
                // A consumed output view does not keep its storage alive in gallocr.
                ggml_set_output(output->view_src);
            }
        }
        for (int node_idx : segment.internal_node_indices) {
            ggml_graph_add_node(segment_graph, ggml_graph_node(gf, node_idx));
        }
        // Fusion eligibility must include consumers outside this segment.
        for (ggml_tensor* tensor : represented) {
            const size_t source_index = ggml_hash_find(&gf->visited_hash_set, tensor);
            if (source_index != GGML_HASHSET_FULL && ggml_bitset_get(gf->visited_hash_set.used, source_index)) {
                const size_t index = ggml_hash_insert(&segment_graph->visited_hash_set, tensor);
                segment_graph->use_counts[index] = gf->use_counts[source_index];
            }
        }
        *graph_ctx_out = graph_ctx;
        return segment_graph;
    }
}
static void segment_check() {
    int cases=0;
    for(bool view:{false,true})for(int outside:{1,2,4}) {
        auto c=ggml_init({4*1024*1024,nullptr,true});auto x=ggml_new_tensor_1d(c,GGML_TYPE_F32,64);
        auto shared=ggml_silu(c,x);auto input=view?ggml_view_1d(c,shared,32,0):shared;
        auto first=ggml_scale(c,input,2.f);auto gf=ggml_new_graph_custom(c,64,false);ggml_build_forward_expand(gf,first);
        for(int j=0;j<outside;++j)ggml_build_forward_expand(gf,ggml_scale(c,input,3.f+j));
        sd::ggml_graph_cut::Segment segment;
        for(int i=0;i<gf->n_nodes;++i)if(gf->nodes[i]==first)segment.internal_node_indices.push_back(i);
        ggml_context* ctx=nullptr;auto cut=sd::ggml_graph_cut::build_segment_graph(gf,segment,&ctx);
        auto a=ggml_hash_find(&gf->visited_hash_set,input),b=ggml_hash_find(&cut->visited_hash_set,input);
        assert(a!=GGML_HASHSET_FULL&&b!=GGML_HASHSET_FULL);assert(gf->use_counts[a]>=outside+1);assert(cut->use_counts[b]==gf->use_counts[a]);
        if(view){auto ai=ggml_hash_find(&gf->visited_hash_set,shared),bi=ggml_hash_find(&cut->visited_hash_set,shared);assert(cut->use_counts[bi]==gf->use_counts[ai]);}
        ggml_free(ctx);ggml_free(c);++cases;
    }
    std::fprintf(stderr,"segment external-use cases passed: %d\n",cases);
}
