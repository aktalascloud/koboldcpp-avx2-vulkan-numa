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
#include "extra-checks.h"
#include "segment-check.inc"

static ggml_tensor * original(ggml_context *c,ggml_tensor *g,ggml_tensor *h) {
    auto a=ggml_scale(c,ggml_tanh(c,ggml_scale(c,g,.25f)),4.f);
    a=ggml_mul(c,a,ggml_sigmoid(c,g));
    auto b=ggml_scale(c,ggml_tanh(c,ggml_scale(c,h,1.f/25.f)),25.f);
    return ggml_mul(c,a,b);
}
struct Graph {
    bool fused=false;
    ggml_context * ctx;ggml_backend_buffer_t buffer;ggml_cgraph * graph;ggml_tensor * out;
    Graph(ggml_backend_t backend,int width,int rows,int channels,int samples,int gap,bool swapped,bool split,bool candidate) {
        ctx=ggml_init({4*1024*1024,nullptr,true});assert(ctx);
        const int pitch=2*width+3;
        const size_t nb1=pitch*4,nb2=nb1*(rows+gap),nb3=nb2*(channels+gap);
        const size_t count=nb3*samples/4+16;
        auto input=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,count);ggml_set_input(input);
        auto view=ggml_view_4d(ctx,input,2*width,rows,channels,samples,nb1,nb2,nb3,4);
        auto a=ggml_view_4d(ctx,view,width,rows,channels,samples,nb1,nb2,nb3,0);
        auto z=ggml_view_4d(ctx,view,width,rows,channels,samples,nb1,nb2,nb3,width*4);
        auto gate=swapped?z:a, hidden=swapped?a:z;
        if(split) {gate=ggml_scale(ctx,ggml_cont(ctx,gate),1.f);hidden=ggml_scale(ctx,ggml_cont(ctx,hidden),1.f);}
        fused=candidate&&ggml_is_contiguous_1(gate)&&ggml_is_contiguous_1(hidden);
        out=fused?ggml_glu_split(ctx,gate,hidden,GGML_GLU_OP_SITU):original(ctx,gate,hidden);
        assert(ggml_backend_supports_op(backend,out));
        ggml_set_output(out);graph=ggml_new_graph_custom(ctx,64,false);ggml_build_forward_expand(graph,out);
        buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);assert(buffer);
        std::vector<float> values(count);
        for(size_t i=0;i<count;++i)values[i]=float(int((i*7919+17)%65521)-32760)/997.f;
        const float edges[]={0.f,-0.f,1e-38f,-1e-38f,80.f,-80.f,1000.f,-1000.f};
        for(size_t i=0;i<count;i+=127)values[i]=edges[(i/127)%8];
        ggml_backend_tensor_set(input,values.data(),0,count*4);
    }
    ~Graph(){ggml_backend_buffer_free(buffer);ggml_free(ctx);}
    void run(ggml_backend_t b){assert(ggml_backend_graph_compute(b,graph)==GGML_STATUS_SUCCESS);}
    std::vector<float> read(){std::vector<float> d(ggml_nelements(out));ggml_backend_tensor_get(out,d.data(),0,d.size()*4);return d;}
};
struct MlpGraph {
    ggml_context *c;ggml_backend_buffer_t buffer;ggml_cgraph *graph;ggml_tensor *out;
    MlpGraph(ggml_backend_t b,int tokens,int chunk) {
        c=ggml_init({4*1024*1024,nullptr,true});assert(c);
        auto x=ggml_new_tensor_2d(c,GGML_TYPE_F32,256,tokens);
        auto w0=ggml_new_tensor_2d(c,GGML_TYPE_F32,256,1024);
        auto w1=ggml_new_tensor_2d(c,GGML_TYPE_F32,512,256);
        std::vector<ggml_tensor*> pieces;
        for(int at=0;at<tokens;at+=chunk) {
            int n=std::min(chunk,tokens-at);
            auto view=ggml_view_2d(c,x,256,n,x->nb[1],at*x->nb[1]);
            auto v=ggml_mul_mat(c,w0,view);
            auto a=ggml_swiglu(c,v);
            auto y=ggml_mul_mat(c,w1,a);pieces.push_back(y);
        }
        out=pieces[0];for(size_t i=1;i<pieces.size();++i)out=ggml_concat(c,out,pieces[i],1);
        graph=ggml_new_graph_custom(c,4096,false);ggml_build_forward_expand(graph,out);
        buffer=ggml_backend_alloc_ctx_tensors(c,b);assert(buffer);
        for(auto t:{x,w0,w1}) {std::vector<float> d(ggml_nelements(t));for(size_t i=0;i<d.size();++i)d[i]=float(int((i*7919+17)%65521)-32760)/327600.f;ggml_backend_tensor_set(t,d.data(),0,d.size()*4);}
    }
    ~MlpGraph(){ggml_backend_buffer_free(buffer);ggml_free(c);}
    std::vector<float> run(ggml_backend_t b){assert(ggml_backend_graph_compute(b,graph)==GGML_STATUS_SUCCESS);std::vector<float>d(ggml_nelements(out));ggml_backend_tensor_get(out,d.data(),0,d.size()*4);return d;}
};
int main(int argc,char**argv) {
    if(argc==3 && std::strcmp(argv[1],"--expert-id")==0)return expert_test(std::atoi(argv[2]));
    auto cpu=ggml_backend_cpu_init();auto gpu=ggml_backend_vk_init(0);assert(cpu&&gpu);ggml_backend_cpu_set_n_threads(cpu,16);
    const bool extra_only=argc==2&&std::strcmp(argv[1],"--extra-only")==0;
    int cases=0,mismatches=0;
    std::printf("{\"backends\":[");bool comma=false;
    if(!extra_only)for(auto backend:{cpu,gpu}) {
        int local=0,bad=0,fused_cases=0,fallback_cases=0;float max_error=0;
        for(int width:{1,3,7,8,31,128,3072}) for(int rows:{1,7}) for(bool swapped:{false,true}) for(bool split:{false,true}) for(int gap:{0,2}) {
            Graph a(backend,width,rows,2,2,gap,swapped,split,false),z(backend,width,rows,2,2,gap,swapped,split,true);
            a.run(backend);z.run(backend);auto x=a.read(),y=z.read();assert(x.size()==y.size());
            if(std::memcmp(x.data(),y.data(),x.size()*4)) {++bad;for(size_t i=0;i<x.size();++i)max_error=std::max(max_error,std::fabs(x[i]-y[i]));}
            ++local;if(z.fused)++fused_cases;else ++fallback_cases;
        }
        cases+=local;mismatches+=bad;std::fprintf(stderr,"SiTU %s: fused=%d fallback=%d\n",ggml_backend_name(backend),fused_cases,fallback_cases);
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
    std::fprintf(stderr,"H3 chunking synthetic F32 checks (forced small buffer limit, not a full model):\n");
    if(argc==2&&std::strcmp(argv[1],"--situ-only")==0){ggml_backend_free(gpu);ggml_backend_free(cpu);return mismatches?2:0;}
    for(auto b:{cpu,gpu})for(int tokens:{17,128})for(int chunk:{1,7,16}) {
        MlpGraph a(b,tokens,tokens),z(b,tokens,chunk);auto x=a.run(b),y=z.run(b);int bad=0;float error=0;
        for(size_t i=0;i<x.size();++i)if(std::memcmp(&x[i],&y[i],4)){++bad;error=std::max(error,std::fabs(x[i]-y[i]));}
        std::fprintf(stderr,"{\"backend\":\"%s\",\"tokens\":%d,\"chunk\":%d,\"non_bit_exact_values\":%d,\"max_abs_error\":%.9g}\n",ggml_backend_name(b),tokens,chunk,bad,error);
    }
    segment_check();rms_probe(gpu);
    ggml_backend_free(gpu);ggml_backend_free(cpu);return 0;
}
