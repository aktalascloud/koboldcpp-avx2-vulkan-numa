#include "ggml.h"
#include "ggml-quants.h"
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
template<class... Args> std::string sd_format(const char *,Args...){return "unsupported conversion";}
#include "scalar-function.inc"
int main() {
    std::array<int32_t,10> input {0,1,-1,127,-128,16777216,16777217,INT32_MIN,INT32_MAX,8};
    std::array<float,10> expected {0,1,-1,127,-128,16777216,16777216,-2147483648.0f,2147483648.0f,8};
    std::array<float,12> output {};output.front()=output.back()=73;
    convert_tensor(input.data(),GGML_TYPE_I32,output.data()+1,GGML_TYPE_F32,2,5,{});
    assert(std::memcmp(output.data()+1,expected.data(),sizeof(expected))==0 && output.front()==73 && output.back()==73);
    convert_tensor(input.data(),GGML_TYPE_I32,output.data()+1,GGML_TYPE_F32,0,5,{});
    assert(output.front()==73 && output.back()==73);
    std::array<int32_t,10> copy {};
    convert_tensor(input.data(),GGML_TYPE_I32,copy.data(),GGML_TYPE_I32,2,5,{});assert(input==copy);
    std::array<float,10> floats {};
    convert_tensor(expected.data(),GGML_TYPE_F32,floats.data(),GGML_TYPE_F32,2,5,{});assert(expected==floats);
    std::array<ggml_fp16_t,2> half {ggml_fp32_to_fp16(0.5f),ggml_fp32_to_fp16(-3.0f)};
    convert_tensor(half.data(),GGML_TYPE_F16,floats.data(),GGML_TYPE_F32,1,2,{});assert(floats[0]==0.5f && floats[1]==-3.0f);
    std::puts("{\"i32_scalar_values\":10,\"control_paths\":4,\"weights_or_models_loaded\":false}");
}
