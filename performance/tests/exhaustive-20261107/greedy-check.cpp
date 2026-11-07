#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include "operation-affinity.h"
#if defined(__AVX2__)
#error "Measure the baseline-x64 llama.o recipe; do not add -march=haswell to this fixture."
#endif
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

using llama_token=int32_t;
constexpr llama_token LLAMA_TOKEN_NULL=-1;
struct llama_token_data {llama_token id;float logit;float p;};
struct llama_token_data_array {llama_token_data* data;size_t size;int64_t selected;bool sorted;};
struct llama_sampler;
using Apply=void(*)(llama_sampler*,llama_token_data_array*);
struct llama_sampler_i {Apply apply;};
struct llama_sampler {llama_sampler_i* iface;void* ctx;int accepted=0;llama_token last=-1;};
struct llama_vocab {int size;};
struct llama_model {llama_vocab vocab;};
struct llama_context {
    llama_model model;
    std::vector<float> raw,logits,probs;
    std::vector<llama_token> ids;
    llama_token token=LLAMA_TOKEN_NULL;
};
struct llama_sampler_chain {std::vector<llama_token_data> cur;llama_sampler* inner;};
static void chain_apply(llama_sampler* s,llama_token_data_array* p) {
    auto c=static_cast<llama_sampler_chain*>(s->ctx);
    c->inner->iface->apply(c->inner,p);
}
static llama_sampler_i llama_sampler_chain_i{chain_apply};
#define GGML_ASSERT(x) assert(x)
#define LLAMA_LOG_DEBUG(...) ((void)0)
static llama_token llama_get_sampled_token_ith(llama_context* c,int32_t){return c->token;}
static const float* llama_get_sampled_probs_ith(llama_context* c,int32_t){return c->probs.empty()?nullptr:c->probs.data();}
static const float* llama_get_sampled_logits_ith(llama_context* c,int32_t){return c->logits.empty()?nullptr:c->logits.data();}
static const llama_token* llama_get_sampled_candidates_ith(llama_context* c,int32_t){return c->ids.empty()?nullptr:c->ids.data();}
static const float* llama_get_logits_ith(llama_context* c,int32_t){return c->raw.data();}
static const llama_model* llama_get_model(llama_context* c){return &c->model;}
static const llama_vocab* llama_model_get_vocab(const llama_model* m){return &m->vocab;}
static int llama_vocab_n_tokens(const llama_vocab* v){return v->size;}
static uint32_t llama_get_sampled_probs_count_ith(llama_context* c,int32_t){return c->probs.size();}
static uint32_t llama_get_sampled_logits_count_ith(llama_context* c,int32_t){return c->logits.size();}
static void llama_sampler_accept(llama_sampler* s,llama_token t){++s->accepted;s->last=t;}
static void llama_sampler_apply(llama_sampler* s,llama_token_data_array* p){s->iface->apply(s,p);}

#include "greedy-functions.inc"
__attribute__((noinline)) llama_token run_before(llama_sampler* s,llama_context* c){return sample_before(s,c,0);}
__attribute__((noinline)) llama_token run_after(llama_sampler* s,llama_context* c){return sample_after(s,c,0);}
static volatile int sink;
static int cases;

static void test(int n,int pattern,int mode,bool chain,bool bench) {
    llama_context c{};c.model.vocab.size=n;
    c.raw.resize(n);
    std::mt19937 rng(789);
    for(float& f:c.raw)f=float(int(rng()%65536)-32768)/8192.f;
    if(pattern==1)std::fill(c.raw.begin(),c.raw.end(),1.f);
    if(pattern==2) {std::fill(c.raw.begin(),c.raw.end(),-INFINITY);c.raw[0]=NAN;}
    if(pattern==3) {for(int i=1;i<n;i+=3)c.raw[i]=NAN;}
    if(pattern==4) {std::fill(c.raw.begin(),c.raw.end(),-INFINITY);c.raw[n/2]=INFINITY;c.raw[n-1]=INFINITY;}
    if(pattern==5) {for(int i=0;i<n;++i)c.raw[i]=(i%2)?0.f:-0.f;}
    if(pattern==6) {std::fill(c.raw.begin(),c.raw.end(),-INFINITY);c.raw[n-1]=-1.f;}
    if(pattern==7) {for(int i=0;i<n;++i)c.raw[i]=float(i%5);}
    if(mode) {
        c.logits=c.raw;c.ids.resize(n);
        for(int i=0;i<n;++i)c.ids[i]=n*2-i;
        if(mode==2)c.probs.assign(n,.125f);
        if(mode==3)c.token=765;
    }
    llama_sampler_i old_i{apply_before},new_i{llama_sampler_greedy_apply};
    llama_sampler old{&old_i,nullptr},neo{&new_i,nullptr};
    llama_sampler_chain a{{},&old},b{{},&neo};
    llama_sampler ca{&llama_sampler_chain_i,&a},cb{&llama_sampler_chain_i,&b};
    auto * sa=chain?&ca:&old;auto * sb=chain?&cb:&neo;
    const auto x=run_before(sa,&c),y=run_after(sb,&c);
    if(x!=y || sa->accepted!=1 || sb->accepted!=1 || sa->last!=sb->last) {
        std::fprintf(stderr,"mismatch n=%d pattern=%d mode=%d chain=%d old=%d new=%d\n",n,pattern,mode,chain,x,y);
        std::abort();
    }
    ++cases;
    if(!bench)return;
    std::printf("sample %d %d %d %d exact",n,pattern,mode,chain);
    for(bool tuned:{false,true,true,false,false,true,true,false}) {
        auto fn=tuned?run_after:run_before;
        auto * s=tuned?sb:sa;
        const auto t=std::chrono::steady_clock::now();
        const int repeat=std::max(96,1000000/n);
        for(int i=0;i<repeat;++i)sink=fn(s,&c);
        const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count()/repeat;
        std::printf(" %c%.4f",tuned?'B':'A',us);
    }
    std::puts("");
}
int main(int argc,char** argv) {
    pin_operation_worker(0);
    std::printf("operation_worker_cpus %zu\n",operation_cores().size());
    for(int n:{1,2,3,4,5,7,8,9,31,256,32000,128000,151936,262144})
        for(int p=0;p<8;++p)for(int mode=0;mode<4;++mode)for(bool chain:{false,true})test(n,p,mode,chain,false);
    llama_token_data_array empty{nullptr,0,-1,false};apply_before(nullptr,&empty);assert(empty.selected==0);
    llama_sampler_greedy_apply(nullptr,&empty);assert(empty.selected==0);
    std::printf("greedy_contract_cases %d\n",cases+2);
    if(argc>1 && std::strcmp(argv[1],"--small")==0) {
        for(int n:{1,40,128})for(int pattern:{0,6})test(n,pattern,0,true,true);
        return 0;
    }
    if(argc>1)for(int n:{32000,128000,151936,262144})
        for(int mode:{0,1,2})for(bool chain:{false,true})test(n,0,mode,chain,true);
}
