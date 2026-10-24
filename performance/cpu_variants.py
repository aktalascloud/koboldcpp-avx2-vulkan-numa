"""Strict-math CPU recipe and temporary staging. LTO only; PGO is unsupported."""
from pathlib import Path
import json,shutil
BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
SRC=BASE.parent
HOT=['ggml.o','ggml-cpu.o','ggml-ops.o','ggml-vec.o','ggml-cpu-cpp.o',
 'ggml-cpu-quants.o','ggml-quants.o','sgemm.o','ggml-backend.o','ggml-backend-meta.o',
 'ggml-repack.o','kcpp-repackmapper.o','kcpp-quantmapper.o','ggml-tiled.o',
 'ggml-tiled-kernel.o','ggml-binops.o','ggml-unops.o','ggml-cpu-traits.o','ggml-threading.o','ggml-alloc.o']

def flags(name,jobs,compiler='winlibs'):
    if name not in ('lto','lto-tuned') or compiler!='winlibs' or not 1<=jobs<=4:
        raise RuntimeError('Production requires WinLibs LTO without PGO,1 to4 workers.')
    lto='-flto='+str(jobs)+' -fuse-linker-plugin'
    text='CFLAGS += -fno-math-errno '+lto+'\nCXXFLAGS += -fno-math-errno '+lto+'\n'
    if name=='lto-tuned':
        tuning='-frename-registers -falign-functions=32 -falign-loops=32'
        text+='CFLAGS += '+tuning+'\nCXXFLAGS += '+tuning+'\nLDFLAGS += '+tuning+'\n'
        policy_path=BASE/'cpu-policy.json'
        policy=json.loads(policy_path.read_text(encoding='utf-8')) if policy_path.exists() else {}
        hot=policy.get('hot_tuning_flags','-fno-ipa-cp-clone')
        if hot not in ('','-fno-ipa-cp-clone'):raise RuntimeError('Unknown hot compiler policy.')
        if hot:
            text+=' '.join(HOT)+': CFLAGS += '+hot+'\n'+' '.join(HOT)+': CXXFLAGS += '+hot+'\nLDFLAGS += '+hot+'\n'
    text+='LDFLAGS += '+lto+'\nggml-vulkan-shaders.o: CXXFLAGS += -fno-lto\n'
    from export_abi import generate
    exports=generate()
    text+='CFLAGS += -D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00\nCXXFLAGS += -D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00\n'
    text+='LDFLAGS += -static -static-libgcc -static-libstdc++\nLDFLAGS += "'+str(exports).replace('\\','/')+'" -Wl,--exclude-all-symbols\n'
    return text,'',lto

def stage(name):
    target=BASE/'variants'/name;target.mkdir(parents=True,exist_ok=True)
    for name in ['koboldcpp_default.dll','koboldcpp.py']:shutil.copy2(SRC/name,target/name)
    for name in ['kcpp_adapters','embd_res']:
        if not (target/name).exists():shutil.copytree(SRC/name,target/name)
    output=target/'kcpp_src/bin';output.mkdir(parents=True,exist_ok=True)
    shutil.copy2(SRC/'kcpp_src/bin/simplecpuinfo.exe',output/'simplecpuinfo.exe')
    return target
