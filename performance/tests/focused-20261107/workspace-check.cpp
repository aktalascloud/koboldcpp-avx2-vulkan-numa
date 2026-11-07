#include "core/compute_workspace.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "stable-diffusion.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <windows.h>

// SD backend discovery and logging are stubbed; both actual backends are GGML CPU.
bool sd_backend_is_cpu(ggml_backend_t b) {return std::string(ggml_backend_name(b)).find("CPU")==0;}
ggml_backend_t sd_backend_cpu_init() {return ggml_backend_cpu_init();}
void log_printf(sd_log_level_t,const char*,int,const char*,...) {}

static bool same(const sd::ComputeWorkspace::Measurement & a,const sd::ComputeWorkspace::Measurement & z) {
    if(a.scheduler!=z.scheduler || a.buffers.size()!=z.buffers.size())return false;
    for(size_t i=0;i<a.buffers.size();++i)if(a.buffers[i].backend!=z.buffers[i].backend || a.buffers[i].bytes!=z.buffers[i].bytes)return false;
    return true;
}
int main() {
    auto * cpu=ggml_backend_cpu_init();auto * extra=ggml_backend_cpu_init();assert(cpu && extra);
    int cases=0,calls=0;
    {
        sd::ComputeWorkspace work(cpu);work.set_extra_backends({extra});
        auto * ctx=ggml_init({4*1024*1024,nullptr,true});assert(ctx);
        auto * input=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,128,3);
        ggml_tensor * node=input;
        for(int i=0;i<512;++i)node=ggml_scale(ctx,node,0.9f);
        ggml_set_output(node);
        auto * graph=ggml_new_graph_custom(ctx,2048,false);ggml_build_forward_expand(graph,node);
        ggml_backend_t placement=nullptr;
        auto external=[&](const ggml_tensor * t){return t==input?placement:nullptr;};
        auto assign=[&](ggml_backend_sched_t,ggml_cgraph*){++calls;};
        auto cold_start=std::chrono::steady_clock::now();
        auto reference=work.measure(graph,1234,external,assign,false);
        auto cold_middle=std::chrono::steady_clock::now();
        auto first=work.measure(graph,1234,external,assign,true);assert(same(first,reference));++cases;
        const double first_uncached_us=std::chrono::duration<double,std::micro>(cold_middle-cold_start).count();
        const double first_cached_miss_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-cold_middle).count();
        int count=calls;assert(same(work.measure(graph,1234,external,assign,true),reference));assert(calls==count);++cases;
        auto check_mutation=[&] {
            int before=calls;auto measured=work.measure(graph,1234,external,assign,true);assert(calls==before+1);
            auto expected=work.measure(graph,1234,external,assign,false);assert(same(measured,expected));
            count=calls;assert(same(work.measure(graph,1234,external,assign,true),expected));assert(calls==count);++cases;
        };
        node->name[0]='z';check_mutation();
        float factor=0.75f;std::memcpy(node->op_params,&factor,sizeof(factor));check_mutation();
        input->name[0]='w';check_mutation();
        placement=extra;check_mutation();
        placement=cpu;check_mutation();
        placement=nullptr;check_mutation();
        ggml_set_output(graph->nodes[128]);check_mutation();
        input->ne[1]=4;input->nb[2]=input->nb[3]=input->nb[1]*4;
        for(int j=0;j<graph->n_nodes;++j) {graph->nodes[j]->ne[1]=4;graph->nodes[j]->nb[2]=graph->nodes[j]->nb[3]=graph->nodes[j]->nb[1]*4;}
        check_mutation();input->nb[1]+=32;input->nb[2]=input->nb[3]=input->nb[1]*4;check_mutation();
        auto i=ggml_hash_find(&graph->visited_hash_set,node);assert(i!=GGML_HASHSET_FULL);
        graph->use_counts[i]++;check_mutation();graph->use_counts[i]--;
        work.release();count=calls;work.measure(graph,1234,external,assign,true);assert(calls==count+1);++cases;
        work.set_extra_backends({});auto direct=work.measure(graph,1234,external,assign,true);
        assert(!direct.scheduler && direct.buffers.size()==1 && direct.buffers[0].bytes==1234);++cases;
        work.set_extra_backends({extra});count=calls;work.measure(graph,1234,external,assign,true);assert(calls==count+1);++cases;
        std::printf("{\"workspace_contract_cases\":%d,\"backends\":\"real CPU, synthetic fallback\",\"graph_nodes\":%d,\"first_uncached_us\":%.9f,\"first_cached_miss_us\":%.9f,\"trials\":[",cases,graph->n_nodes,first_uncached_us,first_cached_miss_us);
        DWORD_PTR allowed=0,system=0;GetProcessAffinityMask(GetCurrentProcess(),&allowed,&system);
        auto old_affinity=SetThreadAffinityMask(GetCurrentThread(),allowed & (~allowed+1));
        bool comma=false;
        for(int repeat=0;repeat<7;++repeat) {
            double times[2];
            for(int pass=0;pass<2;++pass) {
                int cached=(pass+repeat)%2;const auto start=std::chrono::steady_clock::now();
                for(int j=0;j<32;++j) {auto m=work.measure(graph,1234,external,assign,cached);assert(m.scheduler);}
                times[cached]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/32;
            }
            if(comma)std::putchar(',');comma=true;
            std::printf("{\"repeat\":%d,\"uncached_us\":%.9f,\"cached_us\":%.9f}",repeat,times[0],times[1]);
        }
        if(old_affinity)SetThreadAffinityMask(GetCurrentThread(),old_affinity);
        std::puts("]}");ggml_free(ctx);
    }
    ggml_backend_free(extra);ggml_backend_free(cpu);
}
