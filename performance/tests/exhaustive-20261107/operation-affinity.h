#pragma once
#define NOMINMAX
#include <windows.h>
#include <cassert>
#include <vector>

// Only these standalone measurement processes are pinned; runtime settings stay untouched.
static const std::vector<DWORD_PTR>& operation_cores() {
    static const auto cores=[] {
        DWORD_PTR allowed=0,system=0;assert(GetProcessAffinityMask(GetCurrentProcess(),&allowed,&system));
        DWORD bytes=0;GetLogicalProcessorInformationEx(RelationProcessorCore,nullptr,&bytes);
        std::vector<unsigned char> storage(bytes);
        assert(GetLogicalProcessorInformationEx(RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(storage.data()),&bytes));
        std::vector<DWORD_PTR> masks,siblings;
        for(size_t offset=0;offset<bytes;) {
            auto p=reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(storage.data()+offset);
            assert(p->Size);
            for(WORD group=0;group<p->Processor.GroupCount;++group) {
                if(p->Processor.GroupMask[group].Group!=0)continue;
                DWORD_PTR mask=p->Processor.GroupMask[group].Mask&allowed;
                if(mask) {
                    const DWORD_PTR first=mask&(~mask+1);masks.push_back(first);mask^=first;
                    while(mask) {const DWORD_PTR next=mask&(~mask+1);siblings.push_back(next);mask^=next;}
                }
            }
            offset+=p->Size;
        }
        masks.insert(masks.end(),siblings.begin(),siblings.end());
        return masks;
    }();
    return cores;
}
static void pin_operation_worker(int worker) {
    const auto & cores=operation_cores();
    if(size_t(worker)<cores.size())assert(SetThreadAffinityMask(GetCurrentThread(),cores[worker]));
}
