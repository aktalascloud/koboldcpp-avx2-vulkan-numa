struct RopeGraph {
    ggml_context*c;ggml_cgraph*g;ggml_backend_buffer_t buffer;ggml_tensor*out;
    RopeGraph(ggml_backend_t b,bool half,int width,int mode,bool trunc,bool back,bool fused,bool reference) {
        c=ggml_init({4*1024*1024,nullptr,true});assert(c);
        auto x=ggml_new_tensor_4d(c,half?GGML_TYPE_F16:GGML_TYPE_F32,width,3,7,1);auto pos=ggml_new_tensor_1d(c,GGML_TYPE_I32,7);auto weight=ggml_new_tensor_1d(c,GGML_TYPE_F32,width);auto input=x;
        if(fused){input=ggml_mul(c,ggml_rms_norm(c,x,1e-5f),weight);if(reference)ggml_set_output(input);}
        out=back?ggml_rope_ext_back(c,input,pos,nullptr,width,mode,4096,10000.f,.25f,1.f,1.f,32.f,1.f):ggml_rope_ext(c,input,pos,nullptr,width,mode,4096,10000.f,.25f,1.f,1.f,32.f,1.f);
        ggml_rope_set_truncate(out,trunc);g=ggml_new_graph_custom(c,32,false);ggml_build_forward_expand(g,out);buffer=ggml_backend_alloc_ctx_tensors(c,b);assert(buffer);
        std::vector<float>d(ggml_nelements(x));for(size_t i=0;i<d.size();++i)d[i]=float(int((i*7919+17)%65521)-32760)/99700.f;
        if(half){std::vector<ggml_fp16_t>h(d.size());for(size_t i=0;i<d.size();++i)h[i]=ggml_fp32_to_fp16(d[i]);ggml_backend_tensor_set(x,h.data(),0,h.size()*2);}else ggml_backend_tensor_set(x,d.data(),0,d.size()*4);
        int32_t positions[]={0,1,17,127,4096,8191,32767};ggml_backend_tensor_set(pos,positions,0,sizeof(positions));std::vector<float>w(width);for(int i=0;i<width;++i)w[i]=1.f+float(i%7)*.01f;ggml_backend_tensor_set(weight,w.data(),0,w.size()*4);
    }
    ~RopeGraph(){ggml_backend_buffer_free(buffer);ggml_free(c);}
    void run(ggml_backend_t b){assert(ggml_backend_graph_compute(b,g)==GGML_STATUS_SUCCESS);}
    std::vector<float>read(){size_t n=ggml_nelements(out);std::vector<float>d(n);if(out->type==GGML_TYPE_F32)ggml_backend_tensor_get(out,d.data(),0,n*4);else{std::vector<ggml_fp16_t>h(n);ggml_backend_tensor_get(out,h.data(),0,n*2);for(size_t i=0;i<n;++i)d[i]=ggml_fp16_to_fp32(h[i]);}return d;}
};
static bool yarn_check(ggml_backend_t cpu,ggml_backend_t gpu){int cases=0,bad=0,fused_pairs=0,half_fallbacks=0,legacy_half_cases=0;double maxerror=0,max_nmse=0,legacy_half_max_nmse=0;int fused_bad=0;
 for(bool half:{false,true})for(int width:{32,64,128})for(int mode:{0,2})for(bool trunc:{false,true})for(bool back:{false,true}){
    RopeGraph a(cpu,half,width,mode,trunc,back,false,true),z(gpu,half,width,mode,trunc,back,false,false);
    if(half&&!trunc){assert(!ggml_backend_supports_op(gpu,z.out));++half_fallbacks;continue;}
    assert(ggml_backend_supports_op(gpu,z.out));a.run(cpu);z.run(gpu);auto x=a.read(),y=z.read();double err=0,den=0;for(size_t i=0;i<x.size();++i){if(!half)maxerror=std::max(maxerror,std::fabs(double(x[i])-y[i]));err+=double(x[i]-y[i])*(x[i]-y[i]);den+=double(x[i])*x[i];}double nmse=err/std::max(den,1e-30);
    if(half){++legacy_half_cases;legacy_half_max_nmse=std::max(legacy_half_max_nmse,nmse);}else{max_nmse=std::max(max_nmse,nmse);if(nmse>1e-7)++bad;++cases;}
    std::fprintf(stderr,"{\"yarn_half\":%d,\"width\":%d,\"mode\":%d,\"truncate\":%d,\"back\":%d,\"nmse\":%.12g}\n",half,width,mode,trunc,back,nmse);
 }
 for(bool trunc:{false,true})for(int mode:{0,2})for(int width:{64,128}){
    RopeGraph a(gpu,false,width,mode,trunc,false,true,true),z(gpu,false,width,mode,trunc,false,true,false);a.run(gpu);z.run(gpu);auto x=a.read(),y=z.read();if(std::memcmp(x.data(),y.data(),x.size()*4))++fused_bad;++fused_pairs;
 }
 std::fprintf(stderr,"{\"yarn_cases\":%d,\"bad_nmse\":%d,\"max_nmse\":%.12g,\"max_abs_error\":%.12g,\"fused_pairs\":%d,\"fused_non_bit_exact\":%d,\"half_nontruncated_cpu_fallbacks\":%d,\"legacy_truncated_half_observations\":%d,\"legacy_half_max_nmse\":%.12g}\n",cases,bad,max_nmse,maxerror,fused_pairs,fused_bad,half_fallbacks,legacy_half_cases,legacy_half_max_nmse);return !bad&&!fused_bad&&half_fallbacks==12;
}
