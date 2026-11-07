#include "iq4-functions.inc"
#include <array>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <vector>
#include <windows.h>
int main() {
    block_iq4_nl block {};
    std::array<float,34> before {}, after {};
    for (uint32_t scale=0;scale<65536;++scale) {
        block.d=(ggml_fp16_t)scale;
        for(int j=0;j<16;++j) block.qs[j]=uint8_t(j|((15-j)<<4));
        decode_before(&block,before.data()+1,32);decode_after(&block,after.data()+1,32);
        if(std::memcmp(before.data(),after.data(),sizeof(before))) { std::fprintf(stderr,"scale mismatch %u\n",scale);return 1; }
    }
    for(int byte=0;byte<256;++byte) {
        block.d=GGML_FP32_TO_FP16(-0.75f);std::memset(block.qs,byte,16);
        decode_before(&block,before.data()+1,32);decode_after(&block,after.data()+1,32);
        assert(std::memcmp(before.data(),after.data(),sizeof(before))==0);
    }
    DWORD_PTR process_mask=0,system_mask=0;GetProcessAffinityMask(GetCurrentProcess(),&process_mask,&system_mask);
    auto old=SetThreadAffinityMask(GetCurrentThread(),process_mask & (~process_mask+1));
    std::printf("{\"exact_scale_cases\":65536,\"packed_byte_cases\":256,\"rgb_or_model_test\":false,\"trials\":[");
    bool first=true;volatile float sink=0;
    for(int count:{32,160,4096,65536}) {
        std::vector<block_iq4_nl> input(count/32);
        for(size_t i=0;i<input.size();++i) { input[i].d=GGML_FP32_TO_FP16(0.125f);for(int j=0;j<16;++j)input[i].qs[j]=uint8_t(i*7+j*13); }
        std::vector<float> a(count+2,91),b(count+2,91);
        decode_before(input.data(),a.data()+1,count);decode_after(input.data(),b.data()+1,count);
        assert(std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0);
        const int loops=std::max(1000,4000000/count);
        for(int repeat=0;repeat<7;++repeat) {
            double times[2];
            for(int pass=0;pass<2;++pass) {
                int which=(pass+repeat)%2;auto fn=which?decode_after:decode_before;
                auto start=std::chrono::steady_clock::now();
                for(int n=0;n<loops;++n) {fn(input.data(),b.data()+1,count);sink=b[1+(n%count)];}
                times[which]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/loops;
            }
            if(!first)std::putchar(',');first=false;
            std::printf("{\"elements\":%d,\"repeat\":%d,\"before_us\":%.9f,\"after_us\":%.9f}",count,repeat,times[0],times[1]);
        }
    }
    if(old)SetThreadAffinityMask(GetCurrentThread(),old);
    std::puts("]}");(void)sink;
}
