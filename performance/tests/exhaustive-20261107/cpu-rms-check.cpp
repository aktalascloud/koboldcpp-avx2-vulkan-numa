#include "operation-affinity.h"
#include "ggml-cpu-impl.h"
#include "vec.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

bool ggml_are_same_shape(const ggml_tensor * a,const ggml_tensor * b) {
    return std::equal(a->ne,a->ne+4,b->ne);
}
void ggml_abort(const char * file,int line,const char * message,...) {
    std::fprintf(stderr,"abort %s:%d %s\n",file,line,message);std::abort();
}
enum ggml_rms_norm_fuse_op {GGML_RMS_NORM_FUSE_OP_NONE,GGML_RMS_NORM_FUSE_OP_MUL};
#include "cpu-rms-functions.inc"

struct Tensor {
    ggml_tensor t{};
    std::vector<float> data;
    Tensor(int n,int r,int h,int b,int pad) {
        t.type=GGML_TYPE_F32;
        t.ne[0]=n;t.ne[1]=r;t.ne[2]=h;t.ne[3]=b;
        t.nb[0]=sizeof(float);t.nb[1]=(n+pad)*sizeof(float);
        t.nb[2]=t.nb[1]*r;t.nb[3]=t.nb[2]*h;
        data.assign(t.nb[3]*b/sizeof(float),-9123.f);t.data=data.data();
        const float eps=1e-5f;std::memcpy(t.op_params,&eps,sizeof(eps));
    }
};
using Fn=void(*)(const ggml_compute_params*,ggml_tensor*,ggml_tensor*);
template<bool after,bool fused>
void run(const ggml_compute_params * p,ggml_tensor * dst,ggml_tensor * mul) {
    if constexpr(after)rms_after<fused?GGML_RMS_NORM_FUSE_OP_MUL:GGML_RMS_NORM_FUSE_OP_NONE>(p,dst,mul);
    else rms_before<fused?GGML_RMS_NORM_FUSE_OP_MUL:GGML_RMS_NORM_FUSE_OP_NONE>(p,dst,mul);
}
static double parallel(Fn fn,Tensor & dst,Tensor & mul,int nth,int repeat) {
    std::atomic<int> ready{0};std::atomic<bool> go{false};
    std::vector<std::thread> pool;
    for(int ith=0;ith<nth;++ith)pool.emplace_back([&,ith] {
        pin_operation_worker(ith);
        ggml_compute_params p{};p.ith=ith;p.nth=nth;
        ready.fetch_add(1);
        while(!go.load(std::memory_order_acquire))std::this_thread::yield();
        for(int i=0;i<repeat;++i)fn(&p,&dst.t,&mul.t);
    });
    while(ready.load()!=nth)std::this_thread::yield();
    const auto start=std::chrono::steady_clock::now();
    go.store(true,std::memory_order_release);
    for(auto & worker:pool)worker.join();
    return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repeat;
}
static int cases;
static void test(int n,int rows,int heads,int batches,int pad,int nth,bool fused,bool bench) {
    Tensor input(n,rows,heads,batches,pad),weight(n,1,1,1,0);
    Tensor a(n,rows,heads,batches,pad),b(n,rows,heads,batches,pad);
    Tensor ma(n,rows,heads,batches,pad),mb(n,rows,heads,batches,pad);
    std::mt19937 rng(793);
    for(float & x:input.data)x=float(int(rng()%65536)-32768)/8192.f;
    for(float & x:weight.data)x=float(int(rng()%65536)-32768)/16384.f;
    a.t.src[0]=b.t.src[0]=&input.t;
    ma.t.src[0]=&a.t;ma.t.src[1]=&weight.t;
    mb.t.src[0]=&weight.t;mb.t.src[1]=&b.t;
    Fn old=fused?run<false,true>:run<false,false>;
    Fn neo=fused?run<true,true>:run<true,false>;
    parallel(old,a,ma,nth,1);parallel(neo,b,mb,nth,1);
    assert(a.data.size()==b.data.size());
    assert(std::memcmp(a.data.data(),b.data.data(),a.data.size()*sizeof(float))==0);
    assert(std::memcmp(ma.data.data(),mb.data.data(),ma.data.size()*sizeof(float))==0);
    ++cases;
    if(!bench)return;
    std::printf("rms %d %d %d %d %d %d %d exact",n,rows,heads,batches,pad,nth,int(fused));
    const int repeat=std::max(8,std::min(512,(16*1024*1024)/(n*rows*heads*batches)));
    for(bool tuned:{false,true,true,false,false,true,true,false}) {
        const double us=parallel(tuned?neo:old,tuned?b:a,tuned?mb:ma,nth,repeat);
        std::printf(" %c%.4f",tuned?'B':'A',us);
    }
    std::puts("");
}
int main(int argc,char**) {
    std::printf("operation_worker_cpus %zu\n",operation_cores().size());
    for(int n:{1,17,256,4096})for(int rows:{1,3,16})for(int pad:{0,7})
        for(int nth:{1,4,16})for(bool fused:{false,true})test(n,rows,5,2,pad,nth,fused,false);
    std::printf("rms_contract_cases %d\n",cases);
    if(argc>1)for(int n:{4096,16384})for(int rows:{1,3,16})for(bool fused:{false,true})
        test(n,rows,16,2,0,16,fused,true);
}
