#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
#include <windows.h>
#include "ggml-cpu/tiled/tiled-kernel.cpp"
#include "tile-adapted.inc"
static uint32_t random_value(uint32_t&v){v=v*1664525u+1013904223u;return v;}
template<int SUB,int BIAS> static void check(int maxweight,int numk,int seed) {
    auto w=std::make_unique<tiled_tile_src0>();auto a=std::make_unique<tiled_tile_src1>();auto z=std::make_unique<tiled_tile_src1>();uint32_t state=seed;
    for(auto&v:w->q)v=uint8_t(BIAS==128?int(random_value(state)%129)-64+128:random_value(state)%(maxweight+1));
    for(auto&v:w->scales)v=int(random_value(state)%127)-63;
    for(auto&v:w->mins)v=0;
    for(auto&v:w->d)v=float(int(random_value(state)%1000)-500)/997.f;
    for(auto&v:w->dmin)v=0;
    for(auto&v:a->q)v=int8_t(random_value(state)>>24);
    for(auto&v:a->bsums)v=0;
    for(auto&v:a->d)v=float(int(random_value(state)%1000)-500)/997.f;
    *z=*a;tiled_repack_src0<SUB>(w.get(),numk==1?64:16,numk,BIAS,true);
    for(int j=0;j<(numk==1?64:16);j+=16){tiled_repack_src1(a.get(),j,numk,true);probe_pack(z.get(),j,numk,true);}
    for(int i:{0,16,32})for(int j:{0,16,32}) {
        if(numk>1&&(i||j))continue;
        alignas(64) float before[64*64],after[64*64];for(int n=0;n<64*64;++n)before[n]=after[n]=float(n%23-11)*.003f;
        for(int slab=0;slab<numk;++slab) {
            if(numk==1){tiled_run_microtile_avx2<SUB,false,BIAS,1,true>(*w,*a,i,j,numk,slab,before,64);probe_tile_4x8<SUB,false,BIAS,1>(*w,*z,i,j,numk,slab,after,64);}
            else {tiled_run_microtile_avx2<SUB,false,BIAS,0,true>(*w,*a,i,j,numk,slab,before,64);probe_tile_4x8<SUB,false,BIAS,0>(*w,*z,i,j,numk,slab,after,64);}
        }
        assert(!std::memcmp(before,after,sizeof(before)));
    }
    if(seed!=1)return;
    alignas(64)float result[256*256]={};std::printf("{\"subblock\":%d,\"bias\":%d,\"num_k\":%d,\"trials\":[",SUB,BIAS,numk);
    for(int repeat=0;repeat<7;++repeat){double t[2];for(int pass=0;pass<2;++pass){int which=(pass+repeat)%2;auto start=std::chrono::steady_clock::now();for(int k=0;k<1000;++k){if(which){if(numk==1)probe_tile_4x8<SUB,false,BIAS,1>(*w,*z,0,0,numk,0,result,256);else probe_tile_4x8<SUB,false,BIAS,0>(*w,*z,0,0,numk,0,result,256);}else {if(numk==1)tiled_run_microtile_avx2<SUB,false,BIAS,1,true>(*w,*a,0,0,numk,0,result,256);else tiled_run_microtile_avx2<SUB,false,BIAS,0,true>(*w,*a,0,0,numk,0,result,256);}}t[which]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/1000;}
        std::printf("%s{\"before_us\":%.9f,\"after_us\":%.9f}",repeat?",":"",t[0],t[1]);}
    std::printf("]}");volatile float consumed=result[0];(void)consumed;
}
int main(){DWORD_PTR available,system;assert(GetProcessAffinityMask(GetCurrentProcess(),&available,&system));assert(SetThreadAffinityMask(GetCurrentThread(),available&(~available+1)));std::printf("{\"configurations\":240,\"comparisons\":880,\"timings\":[");bool first=true;for(int numk:{1,2,4})for(int seed=1;seed<=20;++seed){if(seed==1&&!first)std::putchar(',');check<16,4>(7,numk,seed);if(seed==1)std::putchar(',');check<16,32>(63,numk,seed);if(seed==1)std::putchar(',');check<16,128>(192,numk,seed);if(seed==1)std::putchar(',');check<32,128>(192,numk,seed);first=false;}std::printf("],\"bit_exact\":true}\n");}
