
#include <cstdint>
#include <cassert>
#include <cstdio>
#include <array>
bool invalid0(uint64_t base,uint64_t total,uint64_t data,uint64_t offset,uint64_t span) { return data < base || data - base > total ||
            span > total - (data - base); }
bool invalid1(uint64_t base,uint64_t total,uint64_t data,uint64_t offset,uint64_t span) { return data < base || data > (base + total) || offset > ((base + total) - data) ||
            span > ((base + total) - data - offset); }
bool invalid2(uint64_t base,uint64_t total,uint64_t data,uint64_t offset,uint64_t span) { return data < base || data > (base + total)
         || offset > ((base + total) - data)
         || span > ((base + total) - data - offset); }
bool invalid3(uint64_t base,uint64_t total,uint64_t data,uint64_t offset,uint64_t span) { return data < base || data > (base + total) ||
            offset > ((base + total) - data) ||
            span > ((base + total) - data - offset); }
bool invalid4(uint64_t base,uint64_t total,uint64_t data,uint64_t offset,uint64_t span) { return data < base || data - base > total ||
        span > total - (data - base); }
int main() {
    using fn=bool(*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t);
    fn checks[]={invalid0,invalid1,invalid2,invalid3,invalid4};
    uint64_t count=0;
    for(uint64_t base:{uint64_t(0),uint64_t(4096),UINT64_MAX-4096})
    for(uint64_t total:{uint64_t(0),uint64_t(1),uint64_t(4096)})
    for(uint64_t data:{uint64_t(0),base-1,base,base+total,base+total+1,UINT64_MAX})
    for(uint64_t offset:{uint64_t(0),uint64_t(1),uint64_t(4096),UINT64_MAX})
    for(uint64_t span:{uint64_t(0),uint64_t(1),uint64_t(4096),UINT64_MAX}) {
        for(int i=0;i<5;++i) {
            const uint64_t off=(i==0||i==4)?0:offset;
            using wide=unsigned __int128;
            const bool valid=data>=base && wide(data)+off<=wide(base)+total && wide(data)+off+span<=wide(base)+total;
            assert(checks[i](base,total,data,off,span)==!valid);++count;
        }
    }
    std::printf("{\"rpc_span_cases\":%llu,\"production_predicates\":5}\n",(unsigned long long)count);
}
