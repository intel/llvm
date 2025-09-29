//==------------ ur_win_proxy_loader.cpp - SYCL standard source file ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//

// On Windows, DLLs loaded dynamically (via LoadLibrary) are not tracked as
// dependencies of the caller in the same way they would be if linked
// statically.
// This can lead to unloading problems, where after main() finishes the OS will
// unload those DLLs from memory, possibly before the caller is done.
// (static var destruction or DllMain() can both occur after)
// The workaround is this proxy_loader. It is statically linked by the SYCL
// library and thus is a real dependency and is not unloaded from memory until
// after SYCL itself is unloaded. It calls LoadLibrary on all the UR adapters
// that SYCL will use during its initialization, which ensures that those
// adapter DLLs are not unloaded until after. Note that this property is not
// transitive. If any of the UR DLLs in turn dynamically load some other DLL
// during their lifecycle there is no guarantee that the "grandchild" won't be
// unloaded early. They would need to employ a similar approach.

#include <cassert>
#include <filesystem>
#include <system_error>

#ifdef _WIN32

#include <Windows.h>
#include <direct.h>
#include <malloc.h>
#include <shlwapi.h>

#endif

#include <iostream>
#include <map>
#include <string>

#include "ur_win_proxy_loader.hpp"

#ifdef _WIN32

// ------------------------------------

// cribbed from sycl/source/detail/os_util.cpp
// TODO: Just inline it.
using OSModuleHandle = intptr_t;
static constexpr OSModuleHandle ExeModuleHandle = -1;
static OSModuleHandle getOSModuleHandle(const void *VirtAddr) {
  HMODULE PhModule;
  DWORD Flag = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
  auto LpModuleAddr = reinterpret_cast<LPCSTR>(VirtAddr);
  if (!GetModuleHandleExA(Flag, LpModuleAddr, &PhModule)) {
    // Expect the caller to check for zero and take
    // necessary action
    return 0;
  }
  if (PhModule == GetModuleHandleA(nullptr))
    return ExeModuleHandle;
  return reinterpret_cast<OSModuleHandle>(PhModule);
}

// cribbed from sycl/source/detail/os_util.cpp
/// Returns an absolute path where the object was found.
static std::wstring getCurrentDSODir() {
  wchar_t Path[MAX_PATH];
  auto Handle = getOSModuleHandle(reinterpret_cast<void *>(&getCurrentDSODir));
  DWORD Ret = GetModuleFileName(
      reinterpret_cast<HMODULE>(ExeModuleHandle == Handle ? 0 : Handle), Path,
      MAX_PATH);
  assert(Ret < MAX_PATH && "Path is longer than MAX_PATH?");
  assert(Ret > 0 && "GetModuleFileName failed");
  (void)Ret;

  BOOL RetCode = PathRemoveFileSpec(Path);
  assert(RetCode && "PathRemoveFileSpec failed");
  (void)RetCode;

  return Path;
}

#ifdef _MSC_VER

#ifdef UR_WIN_PROXY_LOADER_DEBUG_POSTFIX
#define UR_LIBRARY_NAME(NAME) "ur_" #NAME "d.dll"
#else
#define UR_LIBRARY_NAME(NAME) "ur_" #NAME ".dll"
#endif

#else // llvm-mingw

#ifdef UR_WIN_PROXY_LOADER_DEBUG_POSTFIX
#define UR_LIBRARY_NAME(NAME) "libur" #NAME "d.dll"
#else
#define UR_LIBRARY_NAME(NAME) "libur" #NAME ".dll"
#endif

#endif

// ------------------------------------

void *&getDllHandle() {
  static void *dllHandle = nullptr;
  return dllHandle;
}

/// Load the adapter libraries
void preloadLibraries() {
  // Suppress system errors.
  // Tells the system to not display the critical-error-handler message box.
  // Instead, the system sends the error to the calling process.
  // This is crucial for graceful handling of adapters that couldn't be
  // loaded, e.g. due to missing native run-times.
  // Sometimes affects L0 or the unified runtime.
  // TODO: add reporting in case of an error.
  // NOTE: we restore the old mode to not affect user app behavior.
  //
  UINT SavedMode = SetErrorMode(SEM_FAILCRITICALERRORS);
  // Exclude current directory from DLL search path
  if (!SetDllDirectory(L"")) {
    assert(false && "Failed to update DLL search path");
  }

  // this path duplicates sycl/detail/ur.cpp:initializeAdapters
  std::filesystem::path LibSYCLDir(getCurrentDSODir());

  // When searching for dependencies of the adapters limit the
  // list of directories to %windows%\system32 and the directory that contains
  // the loaded DLL (the adapter). This is necessary to avoid loading dlls from
  // current directory and some other directories which are considered unsafe.
  auto loadAdapter = [&](auto adapterName,
                         DWORD flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                       LOAD_LIBRARY_SEARCH_SYSTEM32) {
    auto path = LibSYCLDir / adapterName;
    return LoadLibraryEx(path.wstring().c_str(), NULL, flags);
  };
  // We keep the UR Loader handle so it can be fetched by the runtime, but the
  // adapter libraries themselves won't be used.
  getDllHandle() = loadAdapter(UR_LIBRARY_NAME(loader));
  loadAdapter(UR_LIBRARY_NAME(adapter_opencl));
  loadAdapter(UR_LIBRARY_NAME(adapter_level_zero));
  loadAdapter(UR_LIBRARY_NAME(adapter_level_zero_v2));
  loadAdapter(UR_LIBRARY_NAME(adapter_cuda));
  loadAdapter(UR_LIBRARY_NAME(adapter_hip));
  loadAdapter(UR_LIBRARY_NAME(adapter_native_cpu));
  // Load the Level Zero loader dynamic library to ensure it is loaded during
  // the runtime. This is necessary to avoid the level zero loader from being
  // unloaded prematurely. the Only trusted loader is the one that is loaded
  // from the system32 directory.
  LoadLibraryExW(L"ze_loader.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);

  // Restore system error handling.
  (void)SetErrorMode(SavedMode);
  if (!SetDllDirectory(nullptr)) {
    assert(false && "Failed to restore DLL search path");
  }
}

/// windows_ur.cpp:getURLoaderLibrary() calls this to get the DLL loaded
/// earlier.
__declspec(dllexport) void *getPreloadedURLib() { return getDllHandle(); }

BOOL WINAPI DllMain(HINSTANCE hinstDLL, // handle to DLL module
                    DWORD fdwReason,    // reason for calling function
                    LPVOID lpReserved)  // reserved
{
  bool PrintUrTrace = false;
  static const char *UrTrace = std::getenv("SYCL_UR_TRACE");
  static int UrTraceValue = 0;
  if (UrTrace) {
    try {
      UrTraceValue = std::stoi(UrTrace);
    } catch (...) {
      // ignore malformed SYCL_UR_TRACE
    }
  }

  if (UrTraceValue == -1 || UrTraceValue == 2) { // Means print all UR traces
    PrintUrTrace = true;
  }

  switch (fdwReason) {
  case DLL_PROCESS_ATTACH:
    if (PrintUrTrace)
      std::cout << "---> DLL_PROCESS_ATTACH ur_win_proxy_loader.dll\n"
                << std::endl;

    preloadLibraries();
    break;
  case DLL_PROCESS_DETACH:
    if (PrintUrTrace)
      std::cout << "---> DLL_PROCESS_DETACH ur_win_proxy_loader.dll\n"
                << std::endl;
    break;
  case DLL_THREAD_ATTACH:
    break;
  case DLL_THREAD_DETACH:
    break;
  }
  return TRUE;
}

#endif // WIN32
