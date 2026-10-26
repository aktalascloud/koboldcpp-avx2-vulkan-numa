"""Bundle an independent, portable console EXE using absolute resource paths."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json
import PyInstaller.__main__

BASE = Path(__file__).resolve().parent
SRC = BASE.parent
stamp=datetime.now(timezone(timedelta(hours=6)))
(SRC/'embd_res/kcpp_build.json').write_bytes((json.dumps(dict(build_date=stamp.date().isoformat(),
    built_at=stamp.isoformat(),timezone='Asia/Bishkek'),indent=2)+'\n').encode('utf-8'))
args = ['--noconfirm', '--onefile', '--console', '--collect-all', 'customtkinter',
        '--collect-all', 'jinja2', '--collect-all', 'psutil', '--icon', str(SRC / 'niko.ico'),
        '--distpath', str(BASE / 'candidate'), '--workpath', str(BASE / 'package-work'),
        '--specpath', str(BASE), '--name', 'koboldcpp-nocuda']
for source, target in [
    ('kcpp_src/bin/simplecpuinfo.exe', '.'), ('kcpp_src/bin/aria2c-win.exe', '.'),
    ('kcpp_adapters', 'kcpp_adapters'), ('koboldcpp.py', '.'), ('kcpp_agent.py', '.'),
    ('kcpp_src/json_to_gbnf.py', '.'), ('LICENSE.md', '.'),
    ('embd_res', 'embd_res'),
    ('koboldcpp_default.dll', '.'), ('koboldcpp_vulkan.dll', '.')]:
    args += ['--add-data', str(SRC / source) + ';' + target]
for source in ['ggml/src/GGML_LICENSE','otherarch/acestep/ACESTEP_LICENSE',
               'otherarch/audiocpp/upstream/AUDIOCPP_LICENSE','otherarch/qwen3tts/QWEN3TTS_LICENSE',
               'otherarch/sdcpp/SDCPP_LICENSE','otherarch/ttscpp/TTSCPP_LICENSE','src/LLAMACPP_LICENSE']:
    args += ['--add-data',str(SRC/source)+';vendor_licenses']
args += ['--add-data',str(SRC/'otherarch/audiocpp/upstream/external/cJSON/LICENSE')+';vendor_licenses/cJSON']
args += [str(SRC / 'koboldcpp.py')]
PyInstaller.__main__.run(args)
