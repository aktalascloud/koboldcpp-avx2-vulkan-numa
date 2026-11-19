"""Check Q8_K blocks and time the Haswell maximum selection on Windows.

Run from the fork root after installing the pinned WinLibs compiler:
    py -3.12 performance/tests/check_q8k_haswell.py --cc <path to gcc.exe>
"""
import argparse
import hashlib
import json
import re
import statistics
import subprocess
import tempfile
from pathlib import Path

UPSTREAM = 'ed246f4467157b3e5663feae28e7bf4d6990db39'
SOURCE = 'ggml/src/ggml-quants.c'
HEADER = '''#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-impl.h"
#include <assert.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
'''
BENCH = r'''
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include <windows.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

void control(const float *, block_q8_K *, int64_t);
void candidate(const float *, block_q8_K *, int64_t);
typedef void (*quant_fn)(const float *, block_q8_K *, int64_t);
static volatile quant_fn pick;
static float input[24576+8];
static block_q8_K a[96], b[96];
static uint32_t rng=719522;
static uint32_t random_word(void) { rng ^= rng<<13; rng ^= rng>>17; return rng ^= rng<<5; }
static uint32_t floatbits(float x) { uint32_t u; memcpy(&u,&x,4); return u; }
static void randomize(float *x,int n) {
    for(int j=0; j<n; ++j) x[j]=(float)(int32_t)random_word()*0x1p-26f;
}
static double time_block(quant_fn fn, int width, int reps, double hz) {
    LARGE_INTEGER s,e; pick=fn; QueryPerformanceCounter(&s);
    for(int t=0;t<reps;++t) pick(input+(t&7),a,width);
    QueryPerformanceCounter(&e);
    return (e.QuadPart-s.QuadPart)*1e9/hz/reps;
}
int main(void) {
    int cases=0;
    for(int z=0;z<1200;++z) {
        int n=256*(1+(z%96)); float *x=input+z%8; randomize(x,n);
        switch(z%10) {
            case 0: memset(x,0,n*sizeof(*x)); x[n-1]=-0.0f; break;
            case 1: for (int j=0;j<n;++j) x[j]*=0x1p-140f; break;
            case 2: for (int j=0;j<n;j+=256) { x[j+13]=256.f; x[j+224]=-256.f; } break;
            case 3: for (int j=0;j<n;j+=256) { x[j+71]=-256.f; x[j+72]=256.f; } break;
            case 4: for (int j=0;j<n;j+=256) { x[j+1]=NAN; x[j+64]=INFINITY; } break;
            case 5: for (int j=0;j<n;j+=256) { x[j+14]=-FLT_MAX; x[j+255]=FLT_MAX; } break;
            case 6: for (int j=0;j<n;j+=256) { x[j+255]=-127.f; for(int k=0;k<128;k++) x[j+k]=k+0.5f; } break;
        }
        memset(a,0xa5,sizeof(a)); memset(b,0xa5,sizeof(b));
        control(x,a,n); candidate(x,b,n);
        if(memcmp(a,b,sizeof(a))){
            for(int i=0;i<n/256;i++) if(memcmp(&a[i],&b[i],sizeof(a[i]))){
                fprintf(stderr,"mismatch case %d block %d d %08x vs %08x\n",z,i,floatbits(a[i].d),floatbits(b[i].d));
                return 2;
            }
            return 3;
        }
        ++cases;
    }
    printf("exact %d\n",cases);
    LARGE_INTEGER f; QueryPerformanceFrequency(&f);
    randomize(input,24576+8);
    int widths[]={256,1024,4096,8192,12288,24576};
    for(int i=0;i<6;++i) {
        int width=widths[i], reps=26214400/width;
        time_block(control,width,reps/8,(double)f.QuadPart);
        time_block(candidate,width,reps/8,(double)f.QuadPart);
        for(int t=0;t<12;++t){
            double raw_a,raw_b;
            if(t&1){raw_b=time_block(candidate,width,reps,(double)f.QuadPart);raw_a=time_block(control,width,reps,(double)f.QuadPart);}
            else{raw_a=time_block(control,width,reps,(double)f.QuadPart);raw_b=time_block(candidate,width,reps,(double)f.QuadPart);}
            printf("%d %.3f %.3f\n",width,raw_a,raw_b);
        }
    }
}
'''

def extract(text, name):
    return re.search(r'^'+name+r'\(.*?^\}', text, re.M | re.S).group()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', required=True, type=Path)
    args = parser.parse_args()
    cc = args.cc.resolve(strict=True)
    stock = subprocess.check_output(['git', 'show', f'{UPSTREAM}:{SOURCE}']).decode()
    live = Path(SOURCE).read_text(encoding='utf-8')
    symbol = 'quantize_row_q8_K_ref'
    flags = ['-Iggml/include','-Iggml/src','-Iggml/src/ggml-cpu','-std=c11','-O3',
             '-fno-finite-math-only','-D_GNU_SOURCE','-DNDEBUG','-DGGML_USE_CPU',
             '-DGGML_USE_CPU_REPACK','-DGGML_USE_RPC','-DGGML_USE_LLAMAFILE','-pthread',
             '-march=haswell','-mtune=haswell','-mstackrealign','-fno-math-errno',
             '-flto=4','-fuse-linker-plugin','-frename-registers','-falign-functions=32',
             '-falign-loops=32','-fno-ipa-cp-clone','-D_WIN32_WINNT=0x0A00','-DWINVER=0x0A00']
    bodies = []
    with tempfile.TemporaryDirectory(prefix='kcpp-q8k-') as temp:
        folder = Path(temp)
        sources = []
        for name, text in [('control', stock), ('candidate', live)]:
            body = extract(text, 'void '+symbol)
            bodies.append(hashlib.sha256(body.encode()).hexdigest())
            source = folder/(name+'.c')
            source.write_text(HEADER + extract(text,'static inline int nearest_int')+'\n' +
                              body.replace(symbol, '__attribute__((noinline,noclone)) '+name),
                              encoding='utf-8')
            sources.append(source)
        driver = folder/'bench.c'
        driver.write_text(BENCH, encoding='utf-8')
        exe = folder/'check-q8k.exe'
        subprocess.run([cc,*flags,'-static',*sources,driver,'-o',exe], check=True)
        fixture = hashlib.sha256(exe.read_bytes()).hexdigest()
        result = subprocess.check_output([exe], text=True).splitlines()
    rows = {}
    for line in result[1:]:
        width, a, b = line.split()
        rows.setdefault(int(width), []).append([float(a),float(b)])
    print(json.dumps(dict(upstream=UPSTREAM, fixture_sha256=fixture, function_sha256=bodies,
          matches=int(result[0].split()[1]), ns_pairs=rows,
          paired_speedup={width:statistics.median(a/b for a,b in pairs)
                          for width,pairs in rows.items()}), indent=2))

if __name__ == '__main__':
    main()
