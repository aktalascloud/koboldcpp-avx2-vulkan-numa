#include <cassert>
#include <cstdint>
#include <cinttypes>
#include <cstdio>
#include <memory>
#include <unordered_map>
#include <vector>
#define GGML_MAX_SRC 10
#define LOG_ERROR(...) ((void)0)
struct ggml_tensor {
    ggml_tensor * src[GGML_MAX_SRC] {};
    ggml_tensor * view_src = nullptr;
    size_t view_offs = 0;
    void * buffer = nullptr;
    void * data = nullptr;
};
struct rpc_tensor {
    uint64_t src[GGML_MAX_SRC] {};
    uint64_t view_src = 0;
    size_t view_offs = 0;
    bool invalid = false, bad_data = false;
};
struct ggml_context { std::vector<std::unique_ptr<ggml_tensor>> nodes; };
struct rpc_server {
    ggml_tensor * deserialize_tensor(ggml_context * c, const rpc_tensor * t) {
        if (t->invalid) return nullptr;
        c->nodes.emplace_back(new ggml_tensor);
        if (t->bad_data) c->nodes.back()->data = (void *)1;
        return c->nodes.back().get();
    }
    ggml_tensor * create_node(uint64_t, ggml_context *, const std::unordered_map<uint64_t,const rpc_tensor*>&,
                             std::unordered_map<uint64_t,ggml_tensor*>&);
};
#include "rpc-function.inc"
int main() {
    rpc_server server; ggml_context ctx;
    std::unordered_map<uint64_t,const rpc_tensor*> desc;
    std::unordered_map<uint64_t,ggml_tensor*> nodes;
    constexpr size_t depth = 300000;
    std::vector<rpc_tensor> chain(depth);
    for(size_t i=0;i<depth;++i) { if(i+1<depth) chain[i].src[0]=i+2; desc[i+1]=&chain[i]; }
    auto * root = server.create_node(1,&ctx,desc,nodes); assert(root && nodes.size()==depth);
    for(size_t i=1;i<depth;++i) { assert(root->src[0] == nodes.at(i+1)); root=root->src[0]; }
    assert(root->src[0]==nullptr);
    for (int test=0; test<8; ++test) {
        ctx.nodes.clear(); nodes.clear(); desc.clear();
        rpc_tensor a {}, b {}; a.src[0]=2; a.src[1]=2; a.view_src=2; a.view_offs=17;
        desc[1]=&a; desc[2]=&b;
        if(test==1) b.src[0]=1; // cycle, same memoization semantics
        if(test==2) desc.erase(2);
        if(test==3) a.src[0]=a.src[1]=0, a.view_src=3;
        if(test==4) b.invalid=true;
        if(test==5) b.bad_data=true;
        if(test==6) nodes[2]=server.deserialize_tensor(&ctx,&b), desc.erase(2);
        if(test==7) desc.erase(1);
        auto * p=server.create_node(1,&ctx,desc,nodes);
        if(test>=2 && test!=6) { assert(!p); continue; }
        assert(p && p->src[0]==p->src[1] && p->view_src==p->src[0] && p->view_offs==17);
        if(test==1) assert(p->src[0]->src[0]==p);
        assert(server.create_node(1,&ctx,desc,nodes)==p);
    }
    std::puts("{\"rpc_deep_nodes\":300000,\"rpc_contract_cases\":8,\"deserializer\":\"stub\"}");
}
