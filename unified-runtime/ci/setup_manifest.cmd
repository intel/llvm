@echo off
setlocal enabledelayedexpansion

set MANIFEST_FILE="%1"
if not exist %MANIFEST_FILE% (
    echo Can not find manifest file: %MANIFEST_FILE%
    goto failure
)

set IREPO_CHECK=1
if "%IREPO_BASE_DIR%" == "" (
    set IREPO_BASE_DIR=C:\intel\irepo
)
set IREPO_CMD=%IREPO_BASE_DIR%\irepo.bat


::===============================================================
:check_irepo
    echo Looking for %IREPO_CMD%
    if exist %IREPO_CMD% goto irepo_found

    :: IREPO could be set to a different path with different path structure
    :: If not found explicitly set location of both dir and batch file
    set IREPO_BASE_DIR=C:\intel\irepo
    set IREPO_CMD=%IREPO_BASE_DIR%\irepo.bat

    :: If second time through, exit failure
    if [%IREPO_CHECK%]==[2] goto failure

    set IREPO_CHECK=2
    goto check_git


::===============================================================
:check_git
    echo Checking if git installed
    cmd /c git --version
    if [%ERRORLEVEL%]==[0] (
        goto install_irepo
    ) else (
        echo git not installed
        echo Install git and try again
        goto failure
    )


::===============================================================
:install_irepo
    echo Installing irepo
    if not exist %IREPO_BASE_DIR% mkdir %IREPO_BASE_DIR%
    git clone https://github.intel.com/GSDI/irepo %IREPO_BASE_DIR%
    goto check_irepo


::===============================================================
:irepo_found
    echo irepo found at %IREPO_CMD%


::===============================================================
:init_manifest
    echo %IREPO_CMD% select %MANIFEST_FILE%
    cmd /c %IREPO_CMD% select %MANIFEST_FILE%
    if [%ERRORLEVEL%]==[0] (
        goto sync_manifest
    else (
        goto failure
    )


::===============================================================
:sync_manifest
    echo %IREPO_CMD% sync
    cmd /c %IREPO_CMD% sync
    goto exit


::===============================================================
:exit
    exit /B %ERRORLEVEL%


::===============================================================
:failure
    echo Manifest setup failed. Exiting
    exit /B 1