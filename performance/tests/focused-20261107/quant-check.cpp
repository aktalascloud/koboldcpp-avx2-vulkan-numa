#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml-cpu/quants.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
using dot_fn = void (*)(int,float*,size_t,const void*,size_t,const void*,size_t,int);
struct pair_case { ggml_type type; dot_fn simd, generic; bool k; };
#define PAIR(t,n,k) {GGML_TYPE_##t,ggml_vec_dot_##n,ggml_vec_dot_##n##_generic,k}
int main(int argc, char **) {
    ggml_init_params init {1024*1024,nullptr,true}; auto * ctx=ggml_init(init); assert(ctx); ggml_cpu_init();
    pair_case cases[] = {
        PAIR(Q4_0,q4_0_q8_0,false), PAIR(Q5_0,q5_0_q8_0,false), PAIR(Q8_0,q8_0_q8_0,false),
        PAIR(MXFP4,mxfp4_q8_0,false), PAIR(NVFP4,nvfp4_q8_0,false),
        PAIR(IQ4_NL,iq4_nl_q8_0,false), PAIR(IQ1_S,iq1_s_q8_K,true),
        PAIR(IQ1_M,iq1_m_q8_K,true), PAIR(IQ4_XS,iq4_xs_q8_K,true)};
    int tested=0, failed=0;
    for (auto & c : cases) {
        ggml_quantize_init(c.type);
        for (int n : {256,512,4096}) for(int trial=0;trial<8;++trial) {
            std::vector<float> x(n), y(n), imatrix(n,1.0f);
            for(int i=0;i<n;++i) { x[i]=std::sin(float(i*7+trial*3))*2; y[i]=std::cos(float(i*3+trial*7)); }
            std::vector<uint8_t> xb(ggml_row_size(c.type,n)), yb(ggml_row_size(c.k?GGML_TYPE_Q8_K:GGML_TYPE_Q8_0,n));
            ggml_quantize_chunk(c.type,x.data(),xb.data(),0,1,n,imatrix.data());
            if(c.k) quantize_row_q8_K_ref(y.data(),reinterpret_cast<block_q8_K*>(yb.data()),n);
            else quantize_row_q8_0_ref(y.data(),reinterpret_cast<block_q8_0*>(yb.data()),n);
            if(trial) {
                for(int b=0;b<n/(c.k?256:32);++b) {
                    const int offset=trial==1?0:trial==2?15:trial==3?16:trial==4?30:5;
                    if(c.k) {
                        auto & block=reinterpret_cast<block_q8_K*>(yb.data())[b];
                        block.qs[offset]=-128;
                        for(int j=0;j<16;++j) { int sum=0;for(int k=0;k<16;++k)sum+=block.qs[j*16+k];block.bsums[j]=sum; }
                    } else reinterpret_cast<block_q8_0*>(yb.data())[b].qs[offset]=-128;
                    if(c.type==GGML_TYPE_Q8_0 && trial>=5) {
                        auto & a=reinterpret_cast<block_q8_0*>(xb.data())[b];
                        a.qs[offset]=-128; a.qs[offset+1]=-128;
                        reinterpret_cast<block_q8_0*>(yb.data())[b].qs[offset+1]=-128;
                    }
                }
            }
            float got=0, ref=0;
            c.simd(n,&got,0,xb.data(),0,yb.data(),0,1); c.generic(n,&ref,0,xb.data(),0,yb.data(),0,1);
            const bool ok=std::isfinite(got) && std::fabs(got-ref)<=1e-4f*std::max(std::fabs(ref),1.0f);
            ++tested; if(!ok) { ++failed;std::fprintf(stderr,"type=%s n=%d trial=%d got=%g ref=%g\n",ggml_type_name(c.type),n,trial,got,ref); }
        }
    }
    // Exact Q8 extremes at both halves, the last pair and both block scales.
    for(int block=0;block<2;++block) for(int off:{0,8,16,30}) for(int value:{-128,-1,0,1,127}) {
        block_q8_0 x[2] {},y[2] {};
        x[0].d=ggml_fp32_to_fp16(0.5f);x[1].d=ggml_fp32_to_fp16(1.0f);
        y[0].d=y[1].d=ggml_fp32_to_fp16(1.0f);
        x[block].qs[off]=x[block].qs[off+1]=value;
        y[block].qs[off]=y[block].qs[off+1]=-128;
        float got=0;ggml_vec_dot_q8_0_q8_0(64,&got,0,x,0,y,0,1);
        const float ref=2*value*-128*(block?1.0f:0.5f);++tested;if(got!=ref)++failed;
    }
    std::printf("{\"quant_cases\":%d,\"failed\":%d,\"types\":9,\"isa\":\"AVX2/FMA/F16C\"}\n",tested,failed);
    ggml_free(ctx); return argc>1 && failed ? 1:0;
}
