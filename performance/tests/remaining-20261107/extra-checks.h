#pragma once
#include "raw_compute.h"
#include <cstdlib>
#include <filesystem>
static int expert_test(int expert) {
    ggml_set_abort_callback([](const char* message){std::fprintf(stderr,"%s\n",message);std::fflush(stderr);std::_Exit(86);});
    auto b=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(b,4);
    auto c=ggml_init({4*1024*1024,nullptr,true});
    auto w=ggml_new_tensor_3d(c,GGML_TYPE_F32,256,32,2);
    auto x=ggml_new_tensor_3d(c,GGML_TYPE_F32,256,1,1);
    auto ids=ggml_new_tensor_2d(c,GGML_TYPE_I32,1,1);
    auto out=ggml_mul_mat_id(c,w,x,ids);auto graph=ggml_new_graph_custom(c,32,false);ggml_build_forward_expand(graph,out);
    auto buffer=ggml_backend_alloc_ctx_tensors(c,b);assert(buffer);
    std::vector<float> wd(256*32*2,1.f),xd(256,1.f);
    ggml_backend_tensor_set(w,wd.data(),0,wd.size()*4);ggml_backend_tensor_set(x,xd.data(),0,xd.size()*4);ggml_backend_tensor_set(ids,&expert,0,4);
    assert(ggml_backend_graph_compute(b,graph)==GGML_STATUS_SUCCESS);
    std::vector<float> y(32);ggml_backend_tensor_get(out,y.data(),0,128);
    for(float v:y)assert(v==(expert==-1?0.f:256.f));
    ggml_backend_buffer_free(buffer);ggml_free(c);ggml_backend_free(b);return 0;
}
static void rms_probe(ggml_backend_t b) {
    RawCompute raw;
    for(int width:{512,1024,3072})for(int rows:{1,7}) {
        auto c=ggml_init({4*1024*1024,nullptr,true});
        auto x=ggml_new_tensor_2d(c,GGML_TYPE_F32,width,rows);
        auto w=ggml_new_tensor_1d(c,GGML_TYPE_F32,width);
        auto s=ggml_new_tensor_1d(c,GGML_TYPE_F32,width);
        auto h=ggml_new_tensor_1d(c,GGML_TYPE_F32,width);
        auto weighted=ggml_mul(c,ggml_rms_norm(c,x,1e-5f),w);
        auto y=ggml_add(c,ggml_add(c,weighted,ggml_mul(c,weighted,s)),h);
        auto graph=ggml_new_graph_custom(c,32,false);ggml_build_forward_expand(graph,y);
        auto buffer=ggml_backend_alloc_ctx_tensors(c,b);assert(buffer);
        std::vector<std::vector<float>> values;
        for(auto t:{x,w,s,h}) {std::vector<float>d(ggml_nelements(t));for(size_t i=0;i<d.size();++i)d[i]=float(int((i*7919+17+values.size()*111)%65521)-32760)/997.f;ggml_backend_tensor_set(t,d.data(),0,d.size()*4);values.push_back(d);}
        assert(ggml_backend_graph_compute(b,graph)==GGML_STATUS_SUCCESS);std::vector<float> expected(width*rows);ggml_backend_tensor_get(y,expected.data(),0,expected.size()*4);
        struct {uint32_t ne[4],stride[4],offset[5];float eps;uint32_t ne10;}pc{{uint32_t(width),uint32_t(rows),1,1},{1,uint32_t(width),uint32_t(width*rows),uint32_t(width*rows)},{0,0,0,0,0},1e-5f,uint32_t(width)};
        for(const char* file:{"rms-original.spv","rms-wave.spv"}) {
            auto path=(std::filesystem::path(__FILE__).parent_path()/file).string();
            auto result=raw.run(path.c_str(),values,expected.size(),&pc,sizeof(pc),rows);int bad=0;float error=0;
            for(size_t i=0;i<expected.size();++i)if(std::memcmp(&expected[i],&result[i],4)){++bad;error=std::max(error,std::fabs(expected[i]-result[i]));}
            std::fprintf(stderr,"{\"rms_probe\":\"%s\",\"width\":%d,\"rows\":%d,\"non_bit_exact_values\":%d,\"max_abs_error\":%.9g}\n",file,width,rows,bad,error);
        }
        ggml_backend_buffer_free(buffer);ggml_free(c);
    }
}
