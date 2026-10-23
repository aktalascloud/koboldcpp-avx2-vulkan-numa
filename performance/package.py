"""Bundle an independent, portable console EXE using absolute resource paths."""
from pathlib import Path
import PyInstaller.__main__

BASE = Path(__file__).resolve().parent
SRC = BASE.parent
args = ['--noconfirm', '--onefile', '--console', '--collect-all', 'customtkinter',
        '--collect-all', 'jinja2', '--collect-all', 'psutil', '--icon', str(SRC / 'niko.ico'),
        '--distpath', str(BASE / 'candidate'), '--workpath', str(BASE / 'package-work'),
        '--specpath', str(BASE), '--name', 'koboldcpp-nocuda']
for source, target in [
    ('kcpp_src/bin/simplecpuinfo.exe', '.'), ('kcpp_src/bin/aria2c-win.exe', '.'),
    ('kcpp_adapters', 'kcpp_adapters'), ('koboldcpp.py', '.'), ('kcpp_agent.py', '.'),
    ('kcpp_src/json_to_gbnf.py', '.'), ('LICENSE.md', '.'),
    ('MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md', '.'), ('embd_res', 'embd_res'),
    ('koboldcpp_default.dll', '.'), ('koboldcpp_vulkan.dll', '.')]:
    args += ['--add-data', str(SRC / source) + ';' + target]
args += [str(SRC / 'koboldcpp.py')]
PyInstaller.__main__.run(args)
