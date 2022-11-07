@echo off

:: pyenv.cmd
::
:: This script brings up a python virtual environment. It will install all necessary tools through the manifest
:: file specified in :setup_manifest. It will install all pip modules specified in :setup_python. The only dependency
:: is git.

set _CI_ROOTDIR=%~dp0..\
set _CI_SETUP_DIR=%~dp0
set _CI_TOOLS_DIR=%_CI_ROOTDIR%.deps
set _CI_THIRD_PARTY_DIR=%_CI_ROOTDIR%third_party
set VDIR=%_CI_TOOLS_DIR%\pyvirtualenv
set MANIFEST_FILE=%_CI_THIRD_PARTY_DIR%\windows_docs.yml

::echo _CI_ROOTDIR=%_CI_ROOTDIR%
::echo _CI_SETUP_DIR=%_CI_SETUP_DIR%
::echo _CI_TOOLS_DIR=%_CI_TOOLS_DIR%
::echo _CI_THIRD_PARTY_DIR=%_CI_THIRD_PARTY_DIR%
::echo VDIR=%VDIR%
::echo MANIFEST_FILE=%MANIFEST_FILE%

if "%1" == "up" (
    call :up
    goto :eof
)
if "%1" == "down" (
    call :down %*
    goto :eof
)
if "%1" == "reset" (
    call :reset_python %*
    goto :eof
)
if "%1" == "update_pip" (
    call :update_pip %*
    goto :eof
)
if "%1" == "update_manifest" (
    call :setup_manifest %*
    goto :eof
)

echo usage: call %~nx0 ^<up^|down^|reset^|update_pip^|update_manifest^>
goto :eof


::===============================================================
:: Install manifest file (and supporting tools)
:setup_manifest
pushd %_ROOTDIR%
::echo cmd /C %_CI_SETUP_DIR%setup_manifest.cmd %MANIFEST_FILE%
cmd /C %_CI_SETUP_DIR%setup_manifest.cmd %MANIFEST_FILE%
if errorlevel 1 ( exit /B %ERRORLEVEL% )
popd
goto :eof

::===============================================================
:setup_python
set PYTHON_VERSION=3.6.5-2-win64-full
set _PYDIR=%_CI_TOOLS_DIR%\python
if not exist "%_PYDIR%" (
    call :setup_manifest
)
if errorlevel 1 ( exit /B %ERRORLEVEL% )
setlocal enabledelayedexpansion
if not exist "%VDIR%\Scripts\activate.bat" (
    set PYTHONPATH=%_PYDIR%
    if exist "%VDIR%" (
        rd /s/q %VDIR%
    )
    echo **********************************************
    echo * Generating python virtual environment
    echo **********************************************
    %_PYDIR%\python -m venv "%VDIR%"
    if errorlevel 1 ( exit /B %ERRORLEVEL% )
    call "%VDIR%\Scripts\activate.bat"
    if errorlevel 1 ( exit /B %ERRORLEVEL% )
    call "%VDIR%\Scripts\deactivate.bat"

    call :update_pip
)
endlocal
goto :eof


::===============================================================
:: Updates PIP modules (and PIP itself)
:update_pip
echo Updating python PIP modules...
set VDIR_ACTIVE=
if exist "%VDIR%" (
    call "%VDIR%\Scripts\activate.bat"

    rem Use PIP to install required modules
    set http_proxy=http://proxy-us.intel.com:911
    set ftp_proxy=http://proxy-us.intel.com:911
    set https_proxy=http://proxy-us.intel.com:911
    python -m pip install --upgrade pip

    :: Install PIP modules here
    python -m pip install -r %_CI_THIRD_PARTY_DIR%\requirements.txt

    call "%VDIR%\Scripts\deactivate.bat"

    echo Python PIP modules updated
)
echo Done.
goto :eof


::===============================================================
:: Deletes existing python virtual env.
:reset_python
echo Reseting python...
set VDIR_ACTIVE=
if exist "%VDIR%" (
    call "%VDIR%\Scripts\deactivate.bat"
    rmdir /s /q "%VDIR%
)
echo Done.
goto :eof


::===============================================================
:: Activates python virtual env.
:up
call :setup_python
if exist "%VDIR%\Scripts\activate.bat" (
    call "%VDIR%\Scripts\activate.bat"
    if errorlevel 1 ( exit /B %ERRORLEVEL% )
)
setlocal
echo *** Python virtual environment enabled ***
python %* --version
endlocal
goto :eof


::===============================================================
:: Deactivates python virtual env.
:down
if exist "%VDIR%\Scripts\deactivate.bat" (
    call "%VDIR%\Scripts\deactivate.bat"
    if errorlevel 1 ( exit /B %ERRORLEVEL% )
)
goto :eof
