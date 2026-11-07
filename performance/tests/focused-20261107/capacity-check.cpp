#include <cassert>
#include <cstdio>
#include <string>
#include <cstdint>
struct Capacity {
 bool requires_reclamation=false, fits_value=false;
 size_t required_budget_bytes=10, available_budget_bytes=10;
 bool fits() const {return fits_value;}
};
Capacity fast,last;bool vulkan=true;int queries=0,syncs=0;
Capacity check_capacity(int,int,bool detailed=false) {++queries;return detailed?last:fast;}
bool sd_backend_is(int,const std::string &) {return vulkan;}
void ggml_backend_synchronize(int) {++syncs;}
bool retry() {int request=0,required_states=0,compute_backend=0;
    auto capacity = check_capacity(request, required_states);
    if (!capacity.requires_reclamation && capacity.fits()) {
        return true;
    }
    // Synchronize pending frees once before the existing detailed memory query.
    // This tree has no upstream SD Vulkan cache-clear extension; do not invent one.
    if (sd_backend_is(compute_backend, "Vulkan") &&
        capacity.required_budget_bytes <= capacity.available_budget_bytes) {
        ggml_backend_synchronize(compute_backend);
    }
    capacity = check_capacity(request, required_states, true);
    if (!capacity.requires_reclamation && capacity.fits()) {
        return true;
    }
return false;}
int main() {
 int tests=0;
 for(bool gpu:{false,true}) for(bool first_fit:{false,true})
 for(bool last_fit:{false,true}) for(bool reclamation:{false,true}) {
  vulkan=gpu;fast={reclamation,first_fit,10,10};last={reclamation,last_fit,10,10};queries=syncs=0;
  const bool initial=!reclamation && first_fit;
  assert(retry()==(!reclamation && (first_fit||last_fit)));
  assert(queries==(initial?1:2));assert(syncs==(!initial && gpu?1:0));++tests;
 }
 fast={false,false,20,10};last={false,false,20,10};queries=syncs=0;vulkan=true;
 assert(!retry() && queries==2 && syncs==0);++tests;
 std::printf("{\"capacity_cases\":%d,\"production_tail_extracted\":true,\"backend_memory_queries\":\"stub\"}\n",tests);
}
