"""Keep Python's C ABI explicit: MinGW LTO cannot auto-export internal C++ lambdas."""
from pathlib import Path
import re
BASE=Path(__file__).resolve().parent

def generate():
    src=BASE.parent
    text=(src/'kcpp_src/expose.cpp').read_text(encoding='utf-8')
    names=set(re.findall(r'^ {4,7}(?!return\b)[A-Za-z_][\w \t*]*?\b([A-Za-z_]\w*)\s*\(',text,re.M))
    used=set(re.findall(r'\bhandle\.([A-Za-z_]\w*)', (src/'koboldcpp.py').read_text(encoding='utf-8')))
    missing=used-names
    if missing:raise RuntimeError('C ABI declarations drifted: '+','.join(sorted(missing)))
    path=BASE/'koboldcpp-abi.def'
    path.write_text('EXPORTS\n'+''.join('    '+name+'\n' for name in sorted(names)),encoding='ascii')
    return path

if __name__=='__main__':print(generate().read_text())
