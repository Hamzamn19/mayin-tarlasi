@echo off
setlocal

set ROOT=%~dp0

if "%1"=="stop" goto stop

echo Stopping any existing servers...
for /f "tokens=5" %%a in ('netstat -ano ^| find ":8000 "') do taskkill /f /pid %%a 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| find ":3000 "') do taskkill /f /pid %%a 2>nul
timeout /t 2 /nobreak >nul

echo Starting backend...
start /B "" "%ROOT%backend\venv\Scripts\python.exe" "%ROOT%backend\main.py" > "%TEMP%\backend.log" 2>&1
if errorlevel 1 (
  echo Trying system Python...
  start /B "" python "%ROOT%backend\main.py" > "%TEMP%\backend.log" 2>&1
)
echo   Backend starting... ^(http://localhost:8000^)

echo Starting frontend...
start /B "" cmd /c "cd /d "%ROOT%nextjs-app" && npx next dev -p 3000" > "%TEMP%\frontend.log" 2>&1
echo   Frontend starting... ^(http://localhost:3000^)

echo.
echo Waiting for servers...
timeout /t 8 /nobreak >nul

echo Done. Backend at http://localhost:8000 ^| Frontend at http://localhost:3000
exit /b

:stop
echo Stopping servers...
for /f "tokens=5" %%a in ('netstat -ano ^| find ":8000 "') do taskkill /f /pid %%a 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| find ":3000 "') do taskkill /f /pid %%a 2>nul
echo Servers stopped.
exit /b
