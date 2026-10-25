"""Build the patched checkout. No deployment, publishing or model download."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, sys, time

BASE=Path(__file__).resolve().parent
SRC=BASE.parent
sys.path.insert(0,str(BASE))
import cpu_variants
from cpu_flag_options import extra_flags,recipe as backend_recipe

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
    started=time.perf_counter()
    policy=json.loads((BASE/"cpu-policy.json").read_text(encoding="utf-8"))
    if policy.get("pgo_enabled") is not False or policy.get("variant") not in ("lto","lto-tuned"):
        raise RuntimeError("This release requires LTO without PGO")
    recipe,profile,lto=cpu_variants.flags(policy["variant"],options.jobs,"winlibs")
    vk_host=policy.get('vulkan_host_flags','')
    if vk_host not in ('','-fno-semantic-interposition -fno-ipa-cp-clone'):raise RuntimeError('Unknown Vulkan host policy')
    if vk_host:recipe+='\nggml-vulkan.o: CXXFLAGS += '+vk_host+'\n'
    if profile or not lto:raise RuntimeError("No-PGO/LTO policy not satisfied")
    recipe_path=BASE/'optimized.mk';recipe_path.write_text(recipe,encoding='utf-8')
    cpu_variant=policy.get('cpu_hot_variant','baseline')
    cpu_extra=extra_flags(cpu_variant)
    cpu_recipe_path=BASE/'cpu-optimized.mk'
    cpu_recipe_path.write_text(backend_recipe(recipe,cpu_variants.HOT,cpu_variant),encoding='utf-8')
    environment=os.environ.copy()
    environment['PATH']=str(compiler)+os.pathsep+str(make.parent)+os.pathsep+environment['PATH']
    common=[str(make),'-f','Makefile','CC='+str(compiler/'gcc.exe').replace('\\','/'),
            'CXX='+str(compiler/'g++.exe').replace('\\','/'),'LLAMA_VULKAN=1']
    # Force shader regeneration separately. The generator is capped to4 workers.
    run([*common,'-B','-j1','ggml/src/ggml-vulkan-shaders.cpp'],environment,'shaders.log')
    shader_object=SRC/'ggml-vulkan-shaders.o'
    if shader_object.exists():shader_object.unlink()  # -o must not preserve stale embedded SPIR-V.
    args=[*common,'-B','-f',str(BASE/'haswell.mk').replace('\\','/'),'-f',str(recipe_path).replace('\\','/'),
          '-j'+str(options.jobs),'-o','ggml/src/ggml-vulkan-shaders.cpp','-o','vulkan-shaders-gen']
    run([*args,'koboldcpp_vulkan'],environment,'vulkan.log')
    # Full rebuild also prevents instrumentation objects reaching the CPU link.
    cpu_args=[str(cpu_recipe_path).replace('\\','/') if value==str(recipe_path).replace('\\','/') else value for value in args]
    run([*cpu_args,'koboldcpp_default'],environment,'cpu.log')
    subprocess.run([sys.executable,str(BASE/'package.py')],cwd=SRC,env=environment,check=True)
    exe=BASE/'candidate/koboldcpp-nocuda.exe'
    result={**identity,'exe_sha256':sha(exe),'dll_sha256':{path.name:sha(path) for path in (SRC/'koboldcpp_default.dll',SRC/'koboldcpp_vulkan.dll')},
            'cpu_hot_variant':cpu_variant,'cpu_extra_flags':cpu_extra,'vulkan_hot_variant':'baseline',
            'vulkan_host_flags':vk_host,
            'build_date':json.loads((SRC/'embd_res/kcpp_build.json').read_text(encoding='utf-8'))['build_date'],
            'jobs':options.jobs,'profile':profile,'lto':lto,'seconds':time.perf_counter()-started,
            'accepted':False,'note':'Build success is not correctness/performance acceptance.'}
    (BASE/'results/build.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
