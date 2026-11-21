@echo off
setlocal
set "profile=16K"
if not "%~1"=="" set "profile=%~1"
if /i not "%profile%"=="16K" if /i not "%profile%"=="32K" (
    echo Usage: %~nx0 [16K^|32K] [path-to-koboldcpp-nocuda.exe]
    exit /b 2
)
set "native=%~dp0koboldcpp-nocuda.exe"
if not "%~2"=="" set "native=%~f2"
if not exist "%native%" (
    echo Place the rebuilt koboldcpp-nocuda.exe next to this launcher or pass its path.
    exit /b 2
)
pushd "%~dp0" || exit /b 2
"%native%" --config "Native-%profile%.kcpps"
set "result=%errorlevel%"
popd
exit /b %result%
