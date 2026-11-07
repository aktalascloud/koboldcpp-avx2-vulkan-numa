#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"
#include "ggml-cpu/ggml-cpu-impl.h"
#include "ggml-cpu/simd-mappings.h"
#include "ggml-cpu/ops.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>
#include <type_traits>
#include "old-fwht.inc"
#include "hadamard-tables.inc"
#include "yarn-check.inc"
static uint32_t rng=73;
static uint32_t random_bits(){rng=rng*1664525u+1013904223u;return rng;}
static int fwht_bad=0,fwht_cases=0,nan_payload=0,kron_cases=0,kron_bad=0;
static void fwht_check(){
 for(bool half:{false,true})for(int n:{1,2,4,8,16,32,128,512,4096,12,24,48,96,384,768,20,40,80,640,1280})for(int gap:{0,3})for(int threads:{1,4}){
    auto c=ggml_init({4*1024*1024,nullptr,false});assert(c);size_t ts=half?2:4,nb1=(n+gap)*ts,nb2=nb1*5,nb3=nb2*2;
    auto base=ggml_new_tensor_1d(c,half?GGML_TYPE_F16:GGML_TYPE_F32,nb3*2/ts+8);auto x=ggml_view_4d(c,base,n,5,2,2,nb1,nb2,nb3,ts);
    auto dummy=ggml_new_tensor_1d(c,GGML_TYPE_F32,1),a=ggml_new_tensor_4d(c,GGML_TYPE_F32,n,5,2,2),z=ggml_new_tensor_4d(c,GGML_TYPE_F32,n,5,2,2);a->src[0]=z->src[0]=dummy;a->src[1]=z->src[1]=x;
    if(half){auto v=(ggml_fp16_t*)base->data;for(int64_t i=0;i<ggml_nelements(base);++i)v[i]=ggml_fp32_to_fp16(float(int(random_bits()%2001)-1000)/101.f);}else{auto v=(float*)base->data;for(int64_t i=0;i<ggml_nelements(base);++i)v[i]=float(int(random_bits()%2001)-1000)/101.f;}
    for(int ith=0;ith<threads;++ith){ggml_compute_params params{};params.ith=ith;params.nth=threads;
       if((n&(n-1))==0){if(half)baseline::ggml_compute_forward_fwht_impl<ggml_fp16_t>(&params,a);else baseline::ggml_compute_forward_fwht_impl<float>(&params,a);}
       ggml_compute_forward_fwht(&params,z);
    }
    if((n&(n-1))==0){++fwht_cases;if(std::memcmp(a->data,z->data,ggml_nbytes(a)))++fwht_bad;}
    else {++kron_cases;int f=n%12==0?12:20;int groups=n/f;float maxerror=0;for(int row=0;row<20;++row){auto i3=row/10,i2=(row%10)/5,i1=row%5;auto ptr=(char*)x->data+i1*nb1+i2*nb2+i3*nb3;auto out=(float*)z->data+row*n;for(int k=0;k<n;++k){double sum=0;for(int j=0;j<n;++j){int a0=k/f,b0=j/f;int parity=__builtin_parity(unsigned(a0&b0));float sign=(f==12?H12[j%f][k%f]:H20[j%f][k%f])*(parity?-1.f:1.f);float v=half?ggml_fp16_to_fp32(((ggml_fp16_t*)ptr)[j]):((float*)ptr)[j];sum+=double(v)*sign;}float ref=float(sum/std::sqrt(double(n)));maxerror=std::max(maxerror,std::fabs(out[k]-ref));}}if(maxerror>4e-5f)++kron_bad;}
    ggml_free(c);
 }
 for(int test=0;test<64;++test){
    auto c=ggml_init({1024*1024,nullptr,false});auto h=ggml_new_tensor_1d(c,GGML_TYPE_F32,1),x=ggml_new_tensor_1d(c,GGML_TYPE_F32,32),a=ggml_new_tensor_1d(c,GGML_TYPE_F32,32),z=ggml_new_tensor_1d(c,GGML_TYPE_F32,32);a->src[0]=z->src[0]=h;a->src[1]=z->src[1]=x;
    const uint32_t edge[]={0,0x80000000u,1,0x80000001u,0x7f7fffffu,0xff7fffffu,0x7f800000u,0xff800000u,0x7fc01234u,0xffc05678u};for(int i=0;i<32;++i){uint32_t bits=edge[(i+test)%10];std::memcpy((float*)x->data+i,&bits,4);}ggml_compute_params params{};params.nth=1;baseline::ggml_compute_forward_fwht_impl<float>(&params,a);ggml_compute_forward_fwht(&params,z);bool bad=false;for(int i=0;i<32;++i){float u=((float*)a->data)[i],v=((float*)z->data)[i];if(std::isnan(u)&&std::isnan(v))continue;if(std::memcmp(&u,&v,4))bad=true;}++fwht_cases;if(bad)++fwht_bad;ggml_free(c);
 }
 for(int n:{32,128,512,4096}){
    auto c=ggml_init({4*1024*1024,nullptr,false});auto h=ggml_new_tensor_1d(c,GGML_TYPE_F32,1),x=ggml_new_tensor_2d(c,GGML_TYPE_F32,n,4),out=ggml_new_tensor_2d(c,GGML_TYPE_F32,n,4);out->src[0]=h;out->src[1]=x;for(int i=0;i<n*4;++i)((float*)x->data)[i]=float(i%29)*.01f;ggml_compute_params params{};params.nth=1;
    for(int repeat=0;repeat<7;++repeat){double t[2];for(int pass=0;pass<2;++pass){int which=(pass+repeat)%2;auto start=std::chrono::steady_clock::now();for(int k=0;k<1000;++k){if(which)ggml_compute_forward_fwht(&params,out);else baseline::ggml_compute_forward_fwht_impl<float>(&params,out);}t[which]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/1000;}std::fprintf(stderr,"{\"fwht_n\":%d,\"before_us\":%.9f,\"after_us\":%.9f}\n",n,t[0],t[1]);}ggml_free(c);
 }
}
struct MaskGraph{
 ggml_context*c;ggml_cgraph*g;ggml_backend_buffer_t buffer;ggml_tensor*out;
 MaskGraph(ggml_backend_t b,bool half,int kv,int rows,int mode,int gap,int offset){
    c=ggml_init({8*1024*1024,nullptr,true});assert(c);int n_pick=(mode&2)?41:kv,ntop=mode==0?0:67,extra_n=mode&1?19:0;
    size_t ts=half?2:4,nb1=(kv+gap)*ts,nb2=nb1*2;
    auto source=ggml_new_tensor_1d(c,half?GGML_TYPE_F16:GGML_TYPE_F32,nb2*2/ts+8);auto mask=ggml_view_4d(c,source,kv,2,2,1,nb1,nb2,nb2*2,offset*ts);
    auto ids=ggml_new_tensor_4d(c,GGML_TYPE_I32,ntop,rows,2,2);ggml_tensor * score=nullptr,*blocks=nullptr,*extra=nullptr;
    if(mode&1)score=ggml_new_tensor_4d(c,GGML_TYPE_F32,n_pick,rows,2,2);
    if(mode&2)blocks=ggml_new_tensor_2d(c,GGML_TYPE_I32,5,n_pick);
    if(extra_n)extra=ggml_new_tensor_4d(c,GGML_TYPE_I32,extra_n,rows,2,2);
    out=ggml_top_k_mask(c,mask,ids,score,blocks,extra);g=ggml_new_graph_custom(c,32,false);ggml_build_forward_expand(g,out);bool supported_before=ggml_backend_supports_op(b,out);buffer=ggml_backend_alloc_ctx_tensors(c,b);assert(buffer);assert(supported_before==ggml_backend_supports_op(b,out));
    if(half){std::vector<uint16_t>d(ggml_nelements(source));const uint16_t edge[]={0,0x8000,1,0x7c00,0xfc00,0x7e99,0xfeab,0x3555};for(size_t i=0;i<d.size();++i)d[i]=edge[i%8];ggml_backend_tensor_set(source,d.data(),0,d.size()*2);}else{std::vector<uint32_t>d(ggml_nelements(source));const uint32_t edge[]={0,0x80000000u,1,0x7f800000u,0xff800000u,0x7fc19999u,0xffcabcdeu,0x3f555555u};for(size_t i=0;i<d.size();++i)d[i]=edge[i%8];ggml_backend_tensor_set(source,d.data(),0,d.size()*4);}
    std::vector<int32_t>d(ggml_nelements(ids));for(size_t i=0;i<d.size();++i)d[i]=i%9==0?-1:i%9==1?n_pick+7:int(i%17);if(!d.empty())ggml_backend_tensor_set(ids,d.data(),0,d.size()*4);
    if(score){std::vector<uint32_t>s(ggml_nelements(score));const uint32_t values[]={0,0x80000000u,1,0x80000001u,0x7f800000u,0xff800000u,0x7fc00011u,0xffc00022u};for(size_t i=0;i<s.size();++i)s[i]=values[i%8];ggml_backend_tensor_set(score,s.data(),0,s.size()*4);}
    if(blocks){std::vector<int32_t>v(ggml_nelements(blocks));for(size_t i=0;i<v.size();++i)v[i]=i%11==0?-1:i%11==1?kv+9:int((i*37)%kv);ggml_backend_tensor_set(blocks,v.data(),0,v.size()*4);}
    if(extra){std::vector<int32_t>v(ggml_nelements(extra));for(size_t i=0;i<v.size();++i)v[i]=i%7==0?-1:int(i%5);ggml_backend_tensor_set(extra,v.data(),0,v.size()*4);}
 }
 ~MaskGraph(){ggml_backend_buffer_free(buffer);ggml_free(c);}
 void run(ggml_backend_t b){assert(ggml_backend_graph_compute(b,g)==GGML_STATUS_SUCCESS);}
 std::vector<uint8_t>read(){std::vector<uint8_t>d(ggml_nbytes(out));ggml_backend_tensor_get(out,d.data(),0,d.size());return d;}
};
int main(){fwht_check();auto cpu=ggml_backend_cpu_init(),gpu=ggml_backend_vk_init(0);assert(cpu&&gpu);ggml_backend_cpu_set_n_threads(cpu,16);int cases=0,bad=0,fallback=0;
 for(bool half:{false,true})for(int kv:{2,3,32,128,1024,8192,65536,65538})for(int mode:{0,1,2,3})for(int offset:{0,1})for(int gap:{0,2}){
    MaskGraph a(cpu,half,kv,4,mode,gap,offset),z(gpu,half,kv,4,mode,gap,offset);
    if(!ggml_backend_supports_op(gpu,z.out)){++fallback;continue;}
    a.run(cpu);z.run(gpu);auto x=a.read(),y=z.read();if(x!=y)++bad;++cases;
 }
 std::printf("{\"fwht_power_controls\":%d,\"fwht_non_bit_exact\":%d,\"kron_controls\":%d,\"kron_bad_reference\":%d,\"mask_controls\":%d,\"mask_non_bit_exact\":%d,\"mask_fallbacks\":%d,\"mask_trials\":[",fwht_cases,fwht_bad,kron_cases,kron_bad,cases,bad,fallback);
 if(!bad){MaskGraph a(cpu,false,32768,16,3,0,0),z(gpu,false,32768,16,3,0,0);a.run(cpu);z.run(gpu);for(int repeat=0;repeat<7;++repeat){double t[2];for(int pass=0;pass<2;++pass){int which=(pass+repeat)%2;auto start=std::chrono::steady_clock::now();for(int j=0;j<20;++j)(which?z:a).run(which?gpu:cpu);t[which]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/20;}std::printf("%s{\"cpu_us\":%.9f,\"gpu_us\":%.9f}",repeat?",":"",t[0],t[1]);}}
 std::printf("]}\n");bool yarn_ok=yarn_check(cpu,gpu);ggml_backend_free(gpu);ggml_backend_free(cpu);return fwht_bad||kron_bad||bad||!yarn_ok?2:0;
}
