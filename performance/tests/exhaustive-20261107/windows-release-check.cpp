#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

static int warnings;
#define LLAMA_LOG_WARN(...) (++warnings)
static std::string llama_format_win_err(DWORD error) { return std::to_string(error); }
struct Mapping {
    void * addr;
    size_t size;
#include "windows-release-function.inc"
};

static std::vector<unsigned> resident(void * addr,size_t size,size_t page) {
    std::vector<PSAPI_WORKING_SET_EX_INFORMATION> q(size/page);
    for(size_t i=0;i<q.size();++i)q[i].VirtualAddress=static_cast<char *>(addr)+i*page;
    assert(QueryWorkingSetEx(GetCurrentProcess(),q.data(),DWORD(q.size()*sizeof(q[0]))));
    std::vector<unsigned> values;
    for(const auto & p:q)values.push_back(p.VirtualAttributes.Valid);
    return values;
}
static uint64_t checksum(const void * addr,size_t size) {
    auto p=static_cast<const unsigned char *>(addr);
    uint64_t value=14695981039346656037ull;
    for(size_t i=0;i<size;++i)value=(value^p[i])*1099511628211ull;
    return value;
}
int main() {
    constexpr size_t size=64u<<20;
    const wchar_t * path=L"owned-cpp-mmap-fixture.bin";
    HANDLE file=CreateFileW(path,GENERIC_READ|GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_DELETE,nullptr,
        CREATE_NEW,FILE_ATTRIBUTE_TEMPORARY|FILE_FLAG_DELETE_ON_CLOSE,nullptr);
    assert(file!=INVALID_HANDLE_VALUE);
    std::vector<unsigned char> block(65536);
    for(size_t i=0;i<block.size();++i)block[i]=static_cast<unsigned char>(i);
    for(size_t i=0;i<size;i+=block.size()) {
        DWORD count=0;assert(WriteFile(file,block.data(),DWORD(block.size()),&count,nullptr));
        assert(count==block.size());
    }
    HANDLE map=CreateFileMappingW(file,nullptr,PAGE_READONLY,0,0,nullptr);assert(map);
    void * addr=MapViewOfFile(map,FILE_MAP_READ,0,0,size);assert(addr);
    Mapping view{addr,size};
    SYSTEM_INFO info;GetSystemInfo(&info);const size_t page=info.dwPageSize;
    const auto before_hash=checksum(addr,size);
    const auto before=resident(addr,size,page);
    for(auto range:std::vector<std::pair<size_t,size_t>>{{0,0},{10,9},{size,size+page},
            {std::numeric_limits<size_t>::max(),std::numeric_limits<size_t>::max()},{1,page-1}}) {
        view.unmap_fragment(range.first,range.second);
        assert(resident(addr,size,page)==before);
    }
    view.unmap_fragment(1,size-1);
    const auto after=resident(addr,size,page);
    assert(after.front()==before.front() && after.back()==before.back());
    size_t old_pages=0,new_pages=0;
    for(auto valid:before)old_pages+=valid;
    for(auto valid:after)new_pages+=valid;
    assert(new_pages<=6 && old_pages>=size/page-4);
    const auto after_hash=checksum(addr,size);assert(before_hash==after_hash);
    // Releasing a clamped range extending beyond the view must touch no other mapping.
    view.unmap_fragment(size/2,std::numeric_limits<size_t>::max());
    assert(checksum(addr,size)==before_hash && warnings==0);
    std::printf("{\"resident_before\":%zu,\"resident_after\":%zu,\"bytes\":%zu,"
        "\"boundary_pages_retained\":true,\"unchanged_bytes\":true,\"warnings\":%d,"
        "\"scope\":\"actual patched C++ method, Win32 file mapping\"}\n",old_pages*page,new_pages*page,size,warnings);
    assert(UnmapViewOfFile(addr));assert(CloseHandle(map));assert(CloseHandle(file));
}
