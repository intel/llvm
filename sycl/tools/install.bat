@echo off

set OCL_RT_DIR=%~dp0

echo ###
echo ### 1. Save and update OpenCL.dll available in the system
echo ###
set TMP_FILE=%TEMP%\install.bat.tmp

set INSTALL_ERRORS=0

PowerShell -Command "& {(Get-Command .\OpenCL.dll).FileVersionInfo.FileVersion}" > %TMP_FILE%1
set /p DOWNLOADED_OPENCL_VER= < %TMP_FILE%1
echo Downloaded OpenCL.dll verison: %DOWNLOADED_OPENCL_VER%

IF EXIST C:\Windows\System32\OpenCL.dll (
  PowerShell -Command "& {(Get-Command C:\Windows\System32\OpenCL.dll).FileVersionInfo.FileVersion}" > %TMP_FILE%2
  set /p SYSTEM_OPENCL_VER= < %TMP_FILE%2
  echo System OpenCL.dll version: %SYSTEM_OPENCL_VER%

  PowerShell -Command "& {[version]($Env:SYSTEM_OPENCL_VER) -lt ([version]$Env:DOWNLOADED_OPENCL_VER)}" > %TMP_FILE%3
  set /p NEED_OPENCL_UPGRADE= < %TMP_FILE%3
  set /p NEED_OPENCL_BACKUP= < %TMP_FILE%3
) else (
  echo System OpenCL.dll does not exist.
  set NEED_OPENCL_UPGRADE=True
  set NEED_OPENCL_BACKUP=False
)

echo Need to backup C:\Windows\System32\OpenCL.dll : %NEED_OPENCL_BACKUP%
echo Need to update C:\Windows\System32\OpenCL.dll : %NEED_OPENCL_UPGRADE%
echo.

IF %NEED_OPENCL_BACKUP% == True (
  echo Save system OpenCL.dll: C:\Windows\System32\OpenCL.dll to C:\Windows\System32\OpenCL.dll.%SYSTEM_OPENCL_VER%
  copy /Y C:\Windows\System32\OpenCL.dll C:\Windows\System32\OpenCL.dll.%SYSTEM_OPENCL_VER%
  IF ERRORLEVEL 1 (
    echo !!! Cannot save the original file C:\Windows\System32\OpenCL.dll
    echo !!! Try saving the file manually using File Explorer:
    echo !!!     C:\Windows\System32\OpenCL.dll to C:\Windows\System32\OpenCL.dll.%SYSTEM_OPENCL_VER%
    echo !!! Or run this script as Administrator.
    set INSTALL_ERRORS=1
  ) ELSE (
    echo Copy done.
  )
  echo.
)

IF %NEED_OPENCL_UPGRADE% == True (
  echo Replace C:\Windows\System32\OpenCL.dll with the new downloaded OpenCL.dll

  rem CHANGE THE FILE ATTRIBUTES. OTHERWISE, IT CANNOT BE REPLACED by regular MOVE, DEL, COPY commands.
  PowerShell -Command "& {$acl = Get-Acl C:\Windows\System32\OpenCL.dll; $AccessRule = New-Object System.Security.AccessControl.FileSystemAccessRule(\"Users\",\"FullControl\",\"Allow\"); $acl.SetAccessRule($AccessRule); $acl | Set-Acl c:\Windows\System32\OpenCL.dll; }"


  copy /Y %OCL_RT_DIR%OpenCL.dll C:\Windows\System32\
  IF ERRORLEVEL 1 (
    echo !!! Cannot copy new OpenCL.dll to C:\Windows\System32\OpenCL.dll
    echo !!! Try copying the file manually using File Explorer:
    echo !!!     %OCL_RT_DIR%OpenCL.dll to C:\Windows\System32\
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
REG ADD "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v OCL_ICD_FILENAMES /d "%OCL_ICD_FILENAMES%"
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
  echo Create symbolic link or copy tbb.dll and tbbmalloc.tbb to %OCL_RT_DIR%tbb\ after installation
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
  mklink %OCL_RT_DIR%tbb\tbb.dll %1\tbb.dll
  IF ERRORLEVEL 1 (
    echo !!! Cannot create symbolic link for tbb.dll
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
  echo   1. Save C:\Windows\System32\OpenCL.dll to C:\Windows\System32\OpenCL.dll.%SYSTEM_OPENCL_VER%
  echo   2. Copy %OCL_RT_DIR%OpenCL.dll to C:\Windows\System32\OpenCL.dll
  echo   3. Add/set the environment variable OCL_ICD_FILENAMES to %OCL_RT_DIR%intelocl64.dll
  echo   4. Copy TBB libraries or create symbolic links in %OCL_RT_DIR%tbb.
  echo   5. Add/set the environment variable PATH to %OCL_RT_DIR%tbb
  echo Or try running this batch file as Administrator.
) else (
  echo Installation Done SUCCESSFULLY.
)
echo.

endlocal& ^
set OCL_ICD_FILENAMES=%OCL_RT_DIR%intelocl64.dll
set "PATH=%PATH%;%OCL_RT_DIR%\tbb"
