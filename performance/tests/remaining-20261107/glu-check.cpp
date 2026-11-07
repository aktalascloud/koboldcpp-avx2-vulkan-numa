#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static ggml_tensor * old_packed(ggml_context * c,ggml_tensor * x,bool swapped) {
    const auto n=x->ne[0]/2;
    auto a=ggml_view_4d(c,x,n,x->ne[1],x->ne[2],x->ne[3],x->nb[1],x->nb[2],x->nb[3],0);
    auto b=ggml_view_4d(c,x,n,x->ne[1],x->ne[2],x->ne[3],x->nb[1],x->nb[2],x->nb[3],n*4);
    auto gate=ggml_silu_inplace(c,ggml_cont(c,swapped?b:a));
    return ggml_mul(c,swapped?a:b,gate);
}
static ggml_tensor * new_packed(ggml_context * c,ggml_tensor * x,bool swapped) {
    if(x->type==GGML_TYPE_F32 && ggml_is_contiguous_1(x) && x->ne[0]%2==0)
        return swapped?ggml_swiglu_swapped(c,x):ggml_swiglu(c,x);
    return old_packed(c,x,swapped);
}
struct Graph {
    ggml_context * ctx;ggml_backend_buffer_t buffer;ggml_cgraph * graph;ggml_tensor * out;
    Graph(ggml_backend_t backend,int width,int rows,int channels,int samples,int gap,bool swapped,bool split,bool candidate) {
        ctx=ggml_init({4*1024*1024,nullptr,true});assert(ctx);
        const int pitch=2*width+3;
        const size_t nb1=pitch*4,nb2=nb1*(rows+gap),nb3=nb2*(channels+gap);
        const size_t count=nb3*samples/4+16;
        auto input=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,count);ggml_set_input(input);
        auto view=ggml_view_4d(ctx,input,2*width,rows,channels,samples,nb1,nb2,nb3,4);
        if(!split)out=candidate?new_packed(ctx,view,swapped):old_packed(ctx,view,swapped);
        else {
            auto a=ggml_view_4d(ctx,view,width,rows,channels,samples,nb1,nb2,nb3,0);
            auto z=ggml_view_4d(ctx,view,width,rows,channels,samples,nb1,nb2,nb3,width*4);
            auto gate=ggml_scale(ctx,ggml_cont(ctx,swapped?z:a),1.0f);
            if(candidate) {
                auto up=ggml_scale(ctx,ggml_cont(ctx,swapped?a:z),1.0f);
                out=ggml_swiglu_split(ctx,gate,up);
            } else {
                gate=ggml_silu_inplace(ctx,gate);
                auto up=ggml_scale(ctx,ggml_cont(ctx,swapped?a:z),1.0f);
                out=ggml_mul_inplace(ctx,gate,up);
            }
        }
        ggml_set_output(out);graph=ggml_new_graph_custom(ctx,64,false);ggml_build_forward_expand(graph,out);
        buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);assert(buffer);
        std::vector<float> values(count);
        for(size_t i=0;i<count;++i)values[i]=float(int((i*7919+17)%65521)-32760)/997.f;
        const float edges[]={0.f,-0.f,1e-38f,-1e-38f,80.f,-80.f,1e-20f,-1e-20f};
        for(size_t i=0;i<count;i+=127)values[i]=edges[(i/127)%8];
        ggml_backend_tensor_set(input,values.data(),0,count*4);
    }
    ~Graph(){ggml_backend_buffer_free(buffer);ggml_free(ctx);}
    void run(ggml_backend_t b){assert(ggml_backend_graph_compute(b,graph)==GGML_STATUS_SUCCESS);}
    std::vector<float> read(){std::vector<float> d(ggml_nelements(out));ggml_backend_tensor_get(out,d.data(),0,d.size()*4);return d;}
};
int main() {
    auto cpu=ggml_backend_cpu_init();auto gpu=ggml_backend_vk_init(0);assert(cpu&&gpu);ggml_backend_cpu_set_n_threads(cpu,16);
    int cases=0,mismatches=0;
    std::printf("{\"backends\":[");bool comma=false;
    for(auto backend:{cpu,gpu}) {
        int local=0,bad=0;float max_error=0;
        for(int width:{1,3,7,8,31,128,3072}) for(int rows:{1,7}) for(bool swapped:{false,true}) for(bool split:{false,true}) for(int gap:{0,2}) {
            Graph a(backend,width,rows,2,2,gap,swapped,split,false),z(backend,width,rows,2,2,gap,swapped,split,true);
            a.run(backend);z.run(backend);auto x=a.read(),y=z.read();assert(x.size()==y.size());
            if(std::memcmp(x.data(),y.data(),x.size()*4)) {++bad;for(size_t i=0;i<x.size();++i)max_error=std::max(max_error,std::fabs(x[i]-y[i]));}
            ++local;
        }
        cases+=local;mismatches+=bad;
        if(comma)std::putchar(',');comma=true;
        std::printf("{\"name\":\"%s\",\"cases\":%d,\"non_bit_exact\":%d,\"max_abs_error\":%.9g,\"timings\":[",ggml_backend_name(backend),local,bad,max_error);
        bool first=true;
        if(!bad)for(bool split:{false,true}) {
            Graph a(backend,3072,128,1,1,0,false,split,false),z(backend,3072,128,1,1,0,false,split,true);
            a.run(backend);z.run(backend);
            for(int repeat=0;repeat<7;++repeat) {
                double times[2];for(int pass=0;pass<2;++pass){int which=(pass+repeat)%2;auto & g=which?z:a;auto start=std::chrono::steady_clock::now();for(int j=0;j<20;++j)g.run(backend);times[which]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/20;}
                if(!first)std::putchar(',');first=false;
                std::printf("{\"split\":%s,\"repeat\":%d,\"before_us\":%.9f,\"after_us\":%.9f}",split?"true":"false",repeat,times[0],times[1]);
            }
        }
        std::printf("]}");
    }
    std::printf("],\"cases\":%d,\"non_bit_exact\":%d}\n",cases,mismatches);
    ggml_backend_free(gpu);ggml_backend_free(cpu);return 0;
}
