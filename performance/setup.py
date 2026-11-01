"""One-command build prerequisites, pinned original toolchains; no models."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,urllib.request,venv,zipfile
BASE=Path(__file__).resolve().parent
PINS=[('winlibs','https://github.com/brechtsanders/winlibs_mingw/releases/download/13.2.0posix-18.1.5-11.0.1-msvcrt-r8/winlibs-x86_64-posix-seh-gcc-13.2.0-mingw-w64msvcrt-11.0.1-r8.zip','636865e19b0bcacec55fbbb93f238db4c0ecce72107157329fc3d6a6afc5d1b7','mingw64/bin/gcc.exe'),('stable','https://github.com/skeeto/w64devkit/releases/download/v1.22.0/w64devkit-1.22.0.zip','19cb33da2210770e729da879f90cd6efe03611de7742295ed528c98cc42c8f16','w64devkit/bin/make.exe')]
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build',action='store_true');args=parser.parse_args()
    if os.name!='nt' or sys.version_info[:2]!=(3,12):raise RuntimeError('Use64-bit Python3.12 on Windows; compiler target requires Haswell/AVX2')
    target=BASE/'toolchain';target.mkdir(exist_ok=True)
    for name,url,digest,probe in PINS:
        directory=target/name
        if (directory/probe).is_file():continue
        if directory.exists() and any(directory.iterdir()):raise RuntimeError('Incomplete existing toolchain preserved: '+str(directory))
        archive=target/(name+'.zip.download');print('Downloading original '+name,flush=True);urllib.request.urlretrieve(url,archive)
        with archive.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=digest:raise RuntimeError('Toolchain SHA256 mismatch: '+name)
        directory.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for row in z.infolist():
                path=(directory/row.filename).resolve()
                if not path.is_relative_to(directory.resolve()) or (row.external_attr>>16)&0o170000==0o120000:raise RuntimeError('Unsafe toolchain ZIP')
            z.extractall(directory)
        if not (directory/probe).is_file():raise RuntimeError('Unexpected toolchain layout')
        archive.unlink()
    environment=BASE/'.venv'
    if not (environment/'Scripts/python.exe').is_file():venv.create(environment,with_pip=True)
    python=environment/'Scripts/python.exe'
    subprocess.run([str(python),'-m','pip','install','-r',str(BASE/'requirements-build.txt')],check=True)
    if args.build:subprocess.run([str(python),str(BASE/'build.py'),'--jobs','4'],check=True)
    else:print('Ready: '+str(python)+' performance/build.py --jobs 4')
if __name__=='__main__':main()
