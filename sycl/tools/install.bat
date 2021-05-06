@echo off
setlocal EnableDelayedExpansion
set OCL_RT_DIR=%~dp0

echo ###
echo ### 1. Save and update OpenCL.dll available in the system
echo ###
set TMP_FILE=%TEMP%\install.bat.tmp

set OCL_RT_ENTRY_LIB=%OCL_RT_DIR%intelocl64.dll
IF NOT EXIST %OCL_RT_ENTRY_LIB% (
   set OCL_RT_ENTRY_LIB=%OCL_RT_DIR%intelocl64_emu.dll
)

IF "%OCL_ICD_FILENAMES%" == "" (
  set EXTENDEXISTING=N
) else (
  echo OCL_ICD_FILENAMES is present and contains %OCL_ICD_FILENAMES%
  :USERINPUT
  set /P "EXTENDEXISTING=Should the OpenCL RT extend existing configuration (Y/N): "
)
IF "%EXTENDEXISTING%" == "N" (
  echo Clean up previous configuration
  set OCL_ICD_FILENAMES=%OCL_RT_ENTRY_LIB%
) else (
  IF "%EXTENDEXISTING%" == "Y" (

    set OCL_ICD_FILENAMES=%OCL_ICD_FILENAMES%;%OCL_RT_ENTRY_LIB%
    echo Extend previous configuration to %OCL_ICD_FILENAMES%;%OCL_RT_ENTRY_LIB%
  ) else (
    echo WARNING: Incorrect input %EXTENDEXISTING%. Only Y and N are allowed.
    goto USERINPUT
  )
)


set SYSTEM_OCL_ICD_LOADER=C:\Windows\System32\OpenCL.dll
set NEW_OCL_ICD_LOADER=%OCL_RT_DIR%\OpenCL.dll
set INSTALL_ERRORS=0

PowerShell -Command "& {(Get-Command %NEW_OCL_ICD_LOADER%).FileVersionInfo.FileVersion}" > %TMP_FILE%1
set /p DOWNLOADED_OPENCL_VER= < %TMP_FILE%1

IF EXIST %SYSTEM_OCL_ICD_LOADER% (
  echo %SYSTEM_OCL_ICD_LOADER% is present. Checking version.
  PowerShell -Command "& {(Get-Command %SYSTEM_OCL_ICD_LOADER%).FileVersionInfo.FileVersion}" > %TMP_FILE%2
  set /p SYSTEM_OPENCL_VER=<%TMP_FILE%2

  PowerShell -Command "& {([version]$Env:SYSTEM_OPENCL_VER) -lt ([version]$Env:DOWNLOADED_OPENCL_VER)}" > %TMP_FILE%3
  set /p NEED_OPENCL_UPGRADE= < %TMP_FILE%3
  set /p NEED_OPENCL_BACKUP= < %TMP_FILE%3
) else (
  echo System OpenCL.dll does not exist.
  set NEED_OPENCL_UPGRADE=True
  set NEED_OPENCL_BACKUP=False
)

echo Downloaded OpenCL.dll version: %DOWNLOADED_OPENCL_VER%
echo System OpenCL.dll version: %SYSTEM_OPENCL_VER%

echo Need to backup %SYSTEM_OCL_ICD_LOADER% : %NEED_OPENCL_BACKUP%
echo Need to update %SYSTEM_OCL_ICD_LOADER% : %NEED_OPENCL_UPGRADE%
echo.

IF %NEED_OPENCL_BACKUP% == True (
  echo Save system OpenCL.dll: %SYSTEM_OCL_ICD_LOADER% to %SYSTEM_OCL_ICD_LOADER%.%SYSTEM_OPENCL_VER%
  copy /Y %SYSTEM_OCL_ICD_LOADER% %SYSTEM_OCL_ICD_LOADER%.%SYSTEM_OPENCL_VER%
  IF ERRORLEVEL 1 (
    echo !!! Cannot save the original file C:\Windows\System32\OpenCL.dll
    echo !!! Try saving the file manually using File Explorer:
    echo !!!     %SYSTEM_OCL_ICD_LOADER% to %SYSTEM_OCL_ICD_LOADER%.%SYSTEM_OPENCL_VER%
    echo !!! Or run this script as Administrator.
    set INSTALL_ERRORS=1
  ) ELSE (
    echo Copy done.
  )
  echo.
)

IF %NEED_OPENCL_UPGRADE% == True (
  echo Replace %SYSTEM_OCL_ICD_LOADER% with the new downloaded %NEW_OCL_ICD_LOADER%

  rem CHANGE THE FILE ATTRIBUTES. OTHERWISE, IT CANNOT BE REPLACED by regular MOVE, DEL, COPY commands.
  PowerShell -Command "& {$acl = Get-Acl %SYSTEM_OCL_ICD_LOADER%; $AccessRule = New-Object System.Security.AccessControl.FileSystemAccessRule(\"Users\",\"FullControl\",\"Allow\"); $acl.SetAccessRule($AccessRule); $acl | Set-Acl %SYSTEM_OCL_ICD_LOADER%; }"


  copy /Y %NEW_OCL_ICD_LOADER% %SYSTEM_OCL_ICD_LOADER%
  IF ERRORLEVEL 1 (
    echo !!! Cannot copy %NEW_OCL_ICD_LOADER% to %SYSTEM_OCL_ICD_LOADER%
    echo !!! Try copying the file manually using File Explorer:
    echo !!!     %NEW_OCL_ICD_LOADER% to %SYSTEM_OCL_ICD_LOADER%
    echo !!! Or run this script as Administrator.
    set INSTALL_ERRORS=1
  ) ELSE (
    echo Copy done.
  )
  echo.
) ELSE (
  echo System OpenCL.dll is already new, no need to upgrade it.
)



echo.
echo ###
echo ### 3. Set the environment variable OCL_ICD_FILENAMES to %OCL_ICD_FILENAMES%
echo ###
REG ADD "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /f /v OCL_ICD_FILENAMES /d "%OCL_ICD_FILENAMES%"
IF ERRORLEVEL 1 (
  echo !!! Cannot set the environment variable OCL_ICD_FILENAMES
  set INSTALL_ERRORS=1
)

echo.
echo ###
echo ### 4. Create symbolink links to TBB files in %OCL_RT_DIR%tbb
echo ###
if "%1" == "" (
  echo No TBB libraries path is specified
  echo Create symbolic link or copy tbb12.dll and tbbmalloc.tbb to %OCL_RT_DIR%tbb\ after installation
) else (
  IF EXIST %OCL_RT_DIR%tbb (
    rmdir %OCL_RT_DIR%tbb
  )
  mkdir %OCL_RT_DIR%tbb
  IF ERRORLEVEL 1 (
    echo !!! Cannot create mkdir %OCL_RT_DIR%tbb
    set INSTALL_ERRORS=1
  )
echo on
  mklink %OCL_RT_DIR%tbb\tbbmalloc.dll %1\tbbmalloc.dll
  IF ERRORLEVEL 1 (
    echo !!! Cannot create symbolic link for tbbmalloc.dll
    set INSTALL_ERRORS=1
  )
  mklink %OCL_RT_DIR%tbb\tbb12.dll %1\tbb12.dll
  IF ERRORLEVEL 1 (
    echo !!! Cannot create symbolic link for tbb12.dll
    set INSTALL_ERRORS=1
  )
echo off
)

echo.
echo ###
echo ### 5. Set the environment variable PATH to %OCL_RT_DIR%tbb
echo ###
PowerShell -Command "& { [Environment]::SetEnvironmentVariable(\"Path\", $env:Path + \";%OCL_RT_DIR%tbb\", [EnvironmentVariableTarget]::Machine) }"

IF ERRORLEVEL 1 (
  echo !!! Cannot set the environment variable PATH
  set INSTALL_ERRORS=1
)

IF ERRORLEVEL 1 (
  echo !!! Cannot set the environment variable PATH
  set INSTALL_ERRORS=1
)

del %TMP_FILE%*

echo.
IF %INSTALL_ERRORS% == 1 (
  echo Installation finished WITH ERRORS!
  echo See recommendations printed above and perform the following actions manually:
  echo   1. Save %SYSTEM_OCL_ICD_LOADER% to %SYSTEM_OCL_ICD_LOADER%.%SYSTEM_OPENCL_VER%
  echo   2. Copy %NEW_OCL_ICD_LOADER% to %SYSTEM_OCL_ICD_LOADER%
  echo   3. Add/set the environment variable OCL_ICD_FILENAMES to %OCL_RT_ENTRY_LIB%
  echo   4. Copy TBB libraries or create symbolic links in %OCL_RT_DIR%tbb.
  echo   5. Add/set the environment variable PATH to %OCL_RT_DIR%tbb
  echo Or try running this batch file as Administrator.
) else (
  echo Installation Done SUCCESSFULLY.
)
echo.

endlocal& ^
set OCL_ICD_FILENAMES=%OCL_ICD_FILENAMES%
set "PATH=%PATH%;%OCL_RT_DIR%\tbb"
