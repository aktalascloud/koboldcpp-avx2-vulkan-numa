"""Build the patched checkout. No deployment, publishing or model download."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, sys, time

BASE=Path(__file__).resolve().parent
SRC=BASE.parent
sys.path.insert(0,str(BASE))
import cpu_variants

def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):result.update(block)
    return result.hexdigest()

def run(args,env,log):
    with (BASE/'results'/log).open('w',encoding='utf-8') as stream:
        subprocess.run(args,cwd=SRC,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=int,choices=range(1,5),default=4)
    options=parser.parse_args()
    (BASE/'results').mkdir(exist_ok=True)
    compiler=BASE/'toolchain/winlibs/mingw64/bin'
    make=BASE/'toolchain/stable/w64devkit/bin/make.exe'
    if not all(path.is_file() for path in (compiler/'gcc.exe',compiler/'g++.exe',make)):
        raise RuntimeError('Install the pinned toolchain as described in BUILD.md')
    git=shutil.which('git')
    if not git:raise RuntimeError('Git is required')
    revision=subprocess.check_output([git,'rev-parse','HEAD'],cwd=SRC,text=True).strip()
    # Native file content, not stale build products or .gcda files.
    tracked=subprocess.check_output([git,'ls-files','-z'],cwd=SRC).split(b'\0')
    fingerprint=hashlib.sha256()
    for name in sorted(part.decode('utf-8') for part in tracked if part):
        path=SRC/name
        if path.suffix.lower() in ('.cpp','.c','.h','.hpp','.comp','.glsl','.py','.mk') or name=='Makefile':
            fingerprint.update(name.encode());fingerprint.update(bytes.fromhex(sha(path)))
    compiler_version=subprocess.check_output([str(compiler/'gcc.exe'),'--version'],text=True).splitlines()[0]
    identity={'source_commit':revision,'source_root':str(SRC),'source_patch_sha256':fingerprint.hexdigest(),'compiler':compiler_version}
    metadata_path=BASE/'pgo/profile-metadata.json'
    metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    started=time.perf_counter()
    if any(metadata.get(key)!=value for key,value in identity.items()):
        target=cpu_variants.make('instrument',options.jobs,'winlibs')
        training=cpu_variants.train(target)
        metadata_path.write_text(json.dumps({**identity,'vec_profile_valid':training['vec_profile_valid'],
            'vec_profile_sources_sha256':training['vec_profile_sources_sha256']},indent=2),encoding='utf-8')
    recipe,profile,lto=cpu_variants.flags('pgo-lto-tuned',options.jobs,'winlibs')
    recipe_path=BASE/'optimized.mk';recipe_path.write_text(recipe,encoding='utf-8')
    environment=os.environ.copy()
    environment['PATH']=str(compiler)+os.pathsep+str(make.parent)+os.pathsep+environment['PATH']
    common=[str(make),'-f','Makefile','CC='+str(compiler/'gcc.exe').replace('\\','/'),
            'CXX='+str(compiler/'g++.exe').replace('\\','/'),'LLAMA_VULKAN=1']
    # Force shader regeneration separately. The generator is capped to4 workers.
    run([*common,'-B','-j1','ggml/src/ggml-vulkan-shaders.cpp'],environment,'shaders.log')
    args=[*common,'-B','-f',str(BASE/'haswell.mk').replace('\\','/'),'-f',str(recipe_path).replace('\\','/'),
          '-j'+str(options.jobs),'-o','ggml/src/ggml-vulkan-shaders.cpp','-o','vulkan-shaders-gen']
    run([*args,'koboldcpp_vulkan'],environment,'vulkan.log')
    # Full rebuild also prevents instrumentation objects reaching the CPU link.
    run([*args,'koboldcpp_default'],environment,'cpu.log')
    subprocess.run([sys.executable,str(BASE/'package.py')],cwd=SRC,env=environment,check=True)
    exe=BASE/'candidate/koboldcpp-nocuda.exe'
    result={**identity,'exe_sha256':sha(exe),'dll_sha256':{path.name:sha(path) for path in (SRC/'koboldcpp_default.dll',SRC/'koboldcpp_vulkan.dll')},
            'jobs':options.jobs,'profile':profile,'lto':lto,'seconds':time.perf_counter()-started,
            'accepted':False,'note':'Build success is not correctness/performance acceptance.'}
    (BASE/'results/build.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
