#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
struct Graph {
    ggml_context * c;ggml_backend_buffer_t buffer;ggml_cgraph * graph;ggml_tensor * out,*keep_out=nullptr;
    Graph(ggml_backend_t b,int width,int rows,int channels,int samples,int gap,bool candidate,bool outside,int vector_offset=0) {
        c=ggml_init({8*1024*1024,nullptr,true});assert(c);
        size_t nb1=(width+3)*4,nb2=nb1*(rows+gap),nb3=nb2*(channels+gap);
        auto input=ggml_new_tensor_1d(c,GGML_TYPE_F32,nb3*samples/4+4);
        auto x=ggml_view_4d(c,input,width,rows,channels,samples,nb1,nb2,nb3,4);
        std::vector<ggml_tensor*> sources{input},vectors;
        for(int j=0;j<3;++j){auto src=ggml_new_tensor_1d(c,GGML_TYPE_F32,width+vector_offset+16);sources.push_back(src);vectors.push_back(ggml_view_1d(c,src,width,vector_offset*4));}
        // Visit vector views first, as precomputed conditioning in a real graph.
        graph=ggml_new_graph_custom(c,128,false);for(auto t:vectors)ggml_build_forward_expand(graph,t);
        auto weighted=ggml_mul(c,ggml_rms_norm(c,x,1e-5f),vectors[0]);
        if(!candidate)ggml_set_output(weighted); // retain the ordinary RMS+weight fusion and force the remaining ops
        auto scaled=ggml_mul(c,weighted,vectors[1]);auto added=ggml_add(c,weighted,scaled);out=ggml_add(c,added,vectors[2]);
        ggml_build_forward_expand(graph,out);
        if(outside){keep_out=ggml_scale(c,weighted,.5f);ggml_set_output(keep_out);ggml_build_forward_expand(graph,keep_out);}
        buffer=ggml_backend_alloc_ctx_tensors(c,b);assert(buffer);
        for(size_t j=0;j<sources.size();++j){auto t=sources[j];std::vector<float>d(ggml_nelements(t));for(size_t i=0;i<d.size();++i)d[i]=float(int((i*7919+17+j*111)%65521)-32760)/997.f;for(size_t i=0;i<d.size();i+=127)d[i]=(i/127)%2?-0.f:1e-38f;ggml_backend_tensor_set(t,d.data(),0,d.size()*4);}
    }
    ~Graph(){ggml_backend_buffer_free(buffer);ggml_free(c);}
    void run(ggml_backend_t b){assert(ggml_backend_graph_compute(b,graph)==GGML_STATUS_SUCCESS);}
    std::vector<float> read(){std::vector<float>d(ggml_nelements(out));ggml_backend_tensor_get(out,d.data(),0,d.size()*4);if(keep_out){size_t n=d.size();d.resize(n+ggml_nelements(keep_out));ggml_backend_tensor_get(keep_out,d.data()+n,0,ggml_nbytes(keep_out));}return d;}
};
int main(int argc,char**argv){auto b=ggml_backend_vk_init(0);assert(b);if(argc==2&&std::strcmp(argv[1],"--trace")==0){Graph g(b,3072,7,1,1,0,true,false);g.run(b);ggml_backend_free(b);return 0;}
    int cases=0,bad=0;float error=0;
    for(int width:{256,257,511,512,513,777,1024,1025,3072,4096,4097})for(int rows:{1,7})for(int gap:{0,2})for(bool outside:{false,true})for(int offset:{0,1}) {
        Graph a(b,width,rows,2,2,gap,false,outside,offset),z(b,width,rows,2,2,gap,true,outside,offset);a.run(b);z.run(b);auto x=a.read(),y=z.read();
        if(std::memcmp(x.data(),y.data(),x.size()*4)){++bad;for(size_t i=0;i<x.size();++i)error=std::max(error,std::fabs(x[i]-y[i]));}++cases;
    }
    std::printf("{\"cases\":%d,\"non_bit_exact\":%d,\"max_abs_error\":%.9g,\"trials\":[",cases,bad,error);
    if(!bad){Graph a(b,3072,128,1,1,0,false,false),z(b,3072,128,1,1,0,true,false);a.run(b);z.run(b);
        for(int repeat=0;repeat<7;++repeat){double t[2];for(int pass=0;pass<2;++pass){int which=(repeat+pass)%2;auto start=std::chrono::steady_clock::now();for(int j=0;j<20;++j)(which?z:a).run(b);t[which]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/20;}
            std::printf("%s{\"before_us\":%.9f,\"after_us\":%.9f}",repeat?",":"",t[0],t[1]);}}
    std::printf("]}\n");ggml_backend_free(b);return bad?2:0;}
