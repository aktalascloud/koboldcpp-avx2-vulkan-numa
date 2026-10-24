"""CPU PGO/LTO experiment; the production EXE is never modified here."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
SRC=BASE.parent
DATA=BASE/'pgo/data'
HOT=['ggml.o','ggml-cpu.o','ggml-ops.o','ggml-vec.o','ggml-cpu-cpp.o',
     'ggml-cpu-quants.o','ggml-quants.o','sgemm.o','ggml-backend.o','ggml-backend-meta.o',
     'ggml-repack.o','kcpp-repackmapper.o','kcpp-quantmapper.o','ggml-tiled.o',
     'ggml-tiled-kernel.o','ggml-binops.o','ggml-unops.o','ggml-cpu-traits.o',
     'ggml-threading.o','ggml-alloc.o']


def stage(name):
    target=BASE/'variants'/name
    target.mkdir(parents=True,exist_ok=True)
    shutil.copy2(SRC/'koboldcpp_default.dll',target/'koboldcpp_default.dll')
    shutil.copy2(SRC/'koboldcpp.py',target/'koboldcpp.py')
    for folder in ['kcpp_adapters','embd_res']:
        if not (target/folder).exists():
            shutil.copytree(SRC/folder,target/folder)
    binary=target/'kcpp_src/bin'
    binary.mkdir(parents=True,exist_ok=True)
    shutil.copy2(SRC/'kcpp_src/bin/simplecpuinfo.exe',binary/'simplecpuinfo.exe')
    # Earlier experimental WinLibs DLLs used its POSIX runtime DLL. Stage it locally.
    pthread=BASE/'toolchain/winlibs/mingw64/bin/libwinpthread-1.dll'
    if pthread.is_file():shutil.copy2(pthread,target/'libwinpthread-1.dll')
    return target


def flags(name,jobs,compiler='stable'):
    profile=''
    lto='-flto='+str(jobs)+' -fuse-linker-plugin' if 'lto' in name else ''
    if name=='instrument':
        DATA.mkdir(parents=True,exist_ok=True)
        profile='-fprofile-generate='+str(DATA).replace('\\','/')+' -fprofile-update=atomic'
    elif 'pgo' in name:
        if not list(DATA.rglob('*.gcda')):
            raise RuntimeError('PGO profile is absent; do not call this a PGO build.')
        profile='-fprofile-use='+str(DATA).replace('\\','/')+' -fprofile-correction -fprofile-partial-training -Werror=missing-profile'
    # Whole-program LTO, profile guidance on the actual GGML numerical/scheduler core.
    text='CFLAGS += -fno-math-errno '+lto+'\nCXXFLAGS += -fno-math-errno '+lto+'\n'
    if name in ('pgo-lto-tuned','lto-tuned'):
        tuning='-frename-registers -falign-functions=32 -falign-loops=32'
        text+='CFLAGS += '+tuning+'\nCXXFLAGS += '+tuning+'\nLDFLAGS += '+tuning+'\n'
    if profile:
        text+=' '.join(HOT)+': CFLAGS += '+profile+'\n'
        text+=' '.join(HOT)+': CXXFLAGS += '+profile+'\n'
        text+='LDFLAGS += '+profile+'\n'
        if 'pgo' in name:
            # Old profiles predate the AVX helper fix. Fresh profiles can use vec
            # only when the trained source fingerprints match exactly.
            metadata_path=BASE/'pgo/profile-metadata.json'
            metadata=json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
            vec_paths=['ggml/src/ggml-cpu/vec.cpp','ggml/src/ggml-cpu/vec.h']
            vec_valid=metadata.get('vec_profile_valid') and all(
                metadata.get('vec_profile_sources_sha256',{}).get(path)==hashlib.sha256((SRC/path).read_bytes()).hexdigest()
                for path in vec_paths)
            if not vec_valid:text+='ggml-vec.o: CXXFLAGS += -fno-profile-use\n'
    if name in ('pgo-lto-tuned','lto-tuned'):
        policy_path=BASE/'cpu-policy.json'
        policy=json.loads(policy_path.read_text(encoding='utf-8')) if policy_path.exists() else {}
        hot_flags=policy.get('hot_tuning_flags','')
        if hot_flags not in ['', '-fno-ipa-cp-clone', '-falign-functions=16 -falign-loops=16 -fno-unroll-loops']:
            raise RuntimeError('Unmeasured CPU hot tuning flags.')
        if hot_flags:
            text+=' '.join(HOT)+': CFLAGS += '+hot_flags+'\n'
            text+=' '.join(HOT)+': CXXFLAGS += '+hot_flags+'\n'
            text+='LDFLAGS += '+hot_flags+'\n'
        excluded=policy.get('exclude_pgo_objects',[]) if 'pgo' in name else []
        if any(unit not in HOT for unit in excluded):raise RuntimeError('Unknown PGO exclusion.')
        for unit in excluded:
            text+=unit+': CFLAGS += -fno-profile-use\n'+unit+': CXXFLAGS += -fno-profile-use\n'
    if lto:
        text+='LDFLAGS += '+lto+'\n'
        # This generated object contains shader byte arrays, not executable host code.
        text+='ggml-vulkan-shaders.o: CXXFLAGS += -fno-lto\n'
    if compiler=='winlibs':
        from export_abi import generate
        exports=generate()
        text+='CFLAGS += -D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00\nCXXFLAGS += -D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00\n'
        text+='LDFLAGS += -static -static-libgcc -static-libstdc++\n'
        text+='LDFLAGS += "'+str(exports).replace('\\','/')+'" -Wl,--exclude-all-symbols\n'
    return text,profile,lto


def make(name,jobs,compiler='stable',link_only=False):
    text,profile,lto=flags(name,jobs,compiler)
    recipe=BASE/'cpu-variant.mk'
    recipe.write_text(text,encoding='utf-8')
    env=os.environ.copy()
    compiler_bin=(BASE/'toolchain/winlibs/mingw64/bin') if compiler=='winlibs' else (BASE/'toolchain/stable/w64devkit/bin')
    if not (compiler_bin/'gcc.exe').is_file():raise RuntimeError('Compiler not extracted: '+str(compiler_bin))
    env['PATH']=str(compiler_bin)+os.pathsep+str(BASE/'toolchain/stable/w64devkit/bin')+os.pathsep+env['PATH']
    command=[str(BASE/'toolchain/stable/w64devkit/bin/make.exe'),'-f','Makefile',
             '-f',str(BASE/'haswell.mk').replace('\\','/'),'-f',str(recipe).replace('\\','/'),
             'CC='+str(compiler_bin/'gcc.exe').replace('\\','/'),
             'CXX='+str(compiler_bin/'g++.exe').replace('\\','/'),
             '-j'+str(jobs),'koboldcpp_default']
    started=time.time()
    # Rebuild all core targets when profile flags change. LTO requires every unit.
    if link_only:
        pass
    elif name in ['base-final','lto','pgo-lto']:
        command.insert(1,'-B')
    else:
        for target in HOT:
            command[1:1]=['-W',target.removesuffix('.o')+'.force-profile']
        # A phony dependency forces only the numerical/scheduler core to rebuild.
        recipe.write_text(text+'\n.PHONY: force_cpu_variant\n'+' '.join(HOT)+': force_cpu_variant\n',encoding='utf-8')
        # Remove fake -W arguments: the explicit phony prerequisites are sufficient.
        command=[value for value in command if not value.endswith('.force-profile') and value!='-W']
    log=BASE/'results'/('cpu-build-'+name+'.log')
    with log.open('w',encoding='utf-8') as output:
        subprocess.run(command,cwd=SRC,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
    target=stage(name)
    (target/'recipe.mk').write_text(text,encoding='utf-8')
    row=dict(name=name,seconds=time.time()-started,jobs=jobs,profile=profile,lto=lto,compiler=compiler,
             profile_scope='GGML math, quantization, scheduler and threading core; vec.cpp excluded after AVX inline fix',
             profile_excluded=['ggml-vec.o'] if 'pgo' in name else [],
             dll_sha256=hashlib.sha256((target/'koboldcpp_default.dll').read_bytes()).hexdigest())
    (target/'build.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    # The main build must not reuse instrumented objects with profile-use link flags.
    (BASE/'object-flags.json').write_text(json.dumps(dict(variant=name,compiler=compiler,signature=None)),encoding='utf-8')
    print(json.dumps(row),flush=True)
    return target


def train(target,reset=True,start=0):
    # Never mix counters from changed CFGs or previous instrumented source revisions.
    for old in (list(DATA.rglob('*.gcda')) if reset else []):
        if old.is_symlink() or DATA.resolve() not in old.resolve().parents:
            raise RuntimeError('Unexpected profile path; preserve it.')
        old.unlink()
    model=Path(os.environ['KCPP_TRAIN_MODEL']).resolve()
    if not model.is_file():raise RuntimeError('Set KCPP_TRAIN_MODEL to your existing GGUF')
    prompts=[
        'Describe a professional full-frame photograph of an adult standing by a window in a quiet hotel lobby, natural skin, realistic silk fabric, gentle evening light and accurate neutral colors. '*4,
        'Опиши реалистичную фотографию взрослой женщины в вечернем шёлковом платье в холле отеля. Естественные глаза, фактура кожи и ткани, видимые руки и туфли, тёплый свет лампы и холодный свет ночного окна. Сохрани точные цвета и правдоподобные детали. '*3,
        'Write a clear Python function that sorts a list of dictionaries by date and then name. Explain stable ordering, missing values and how to avoid mutating the input. '*4]
    for i,prompt in enumerate(prompts):
        if i<start:continue
        args=[sys.executable,str(target/'koboldcpp.py'),'--model',str(model),'--usecpu',
              '--contextsize','512','--threads','1','--blasthreads','1','--batchsize','256',
              '--genlimit','32','--skiplauncher','--prompt',prompt,'--quiet']
        with (BASE/'results'/('pgo-training-'+str(i)+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    moe=BASE/'test-models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf'
    if moe.is_file() and not moe.with_suffix(moe.suffix+'.aria2').exists():
        args=[sys.executable,str(target/'koboldcpp.py'),'--model',str(moe),'--usecpu',
              '--contextsize','512','--threads','1','--blasthreads','1','--batchsize','256',
              '--genlimit','8','--skiplauncher','--prompt','Explain why rain falls from clouds in four clear sentences. '*8,'--quiet']
        with (BASE/'results/pgo-training-moe.log').open('w',encoding='utf-8') as log:
            subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    files=list(DATA.rglob('*.gcda'))
    result=dict(profile_files=len(files),bytes=sum(p.stat().st_size for p in files),
                training_prompts=prompts,training_genlimit=32,training_threads=1,model_unchanged=True,
                vec_profile_valid=True,vec_profile_sources_sha256={path:hashlib.sha256((SRC/path).read_bytes()).hexdigest()
                    for path in ['ggml/src/ggml-cpu/vec.cpp','ggml/src/ggml-cpu/vec.h']})
    (BASE/'results/pgo-training.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if len(files)<5:
        raise RuntimeError('No usable GCC profile was written at normal DLL shutdown.')
    print(json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['base-final','instrument','pgo','lto','lto-tuned','pgo-lto','pgo-lto-tuned','stage-base'])
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--compiler',choices=['stable','winlibs'],default='stable')
    parser.add_argument('--link-only',action='store_true')
    options=parser.parse_args()
    if not 1<=options.jobs<=4:raise RuntimeError('WHEA guard: use 1 to 4 compile jobs.')
    if options.mode=='stage-base':
        stage('base');return
    target=make(options.mode,options.jobs,options.compiler,options.link_only)
    if options.mode=='instrument':train(target)


if __name__=='__main__':
    main()
