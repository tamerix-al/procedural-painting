@echo off
setlocal
cd /d "%~dp0"
echo [Primitive] Checking Go installation...
where go >nul 2>nul
if errorlevel 1 (
  echo Go is not installed or not in PATH.
  echo Install Go from https://go.dev/dl/ and RESTART VS Code.
  pause
  exit /b 1
)
echo [Primitive] Installing original fogleman/primitive using Go...
echo Internet is required for the first installation.
go install github.com/fogleman/primitive@latest
if errorlevel 1 (
  echo Failed to install Primitive. Check the messages above.
  echo You can also manually place primitive.exe into tools\
  pause
  exit /b 1
)
for /f "delims=" %%G in ('go env GOPATH') do set "GO_PATH=%%G"
if not exist "%GO_PATH%\bin\primitive.exe" (
  echo primitive.exe not found at %GO_PATH%\bin\primitive.exe
  pause
  exit /b 1
)
if not exist tools mkdir tools
copy /Y "%GO_PATH%\bin\primitive.exe" "tools\primitive.exe" >nul
if errorlevel 1 (
  echo Cannot copy executable into tools\
  pause
  exit /b 1
)
echo.
echo Primitive installed to tools\primitive.exe
pause
endlocal
