//==------------ pi_win_proxy_loader.cpp - SYCL standard source file ------==//
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
// after SYCL itself is unloaded. It calls LoadLibrary on all the PI Plugins
// that SYCL will use during its initialization, which ensures that those plugin
// DLLs are not unloaded until after.
// Note that this property is not transitive. If any of the PI DLLs in turn
// dynamically load some other DLL during their lifecycle there is no guarantee
// that the "grandchild" won't be unloaded early. They would need to employ a
// similar approach.

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

#include "pi_win_proxy_loader.hpp"

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
std::wstring getCurrentDSODir() {
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

// these are cribbed from include/sycl/detail/pi.hpp
// a new plugin must be added to both places.
#ifdef _MSC_VER
#define __SYCL_OPENCL_PLUGIN_NAME "pi_opencl.dll"
#define __SYCL_LEVEL_ZERO_PLUGIN_NAME "pi_level_zero.dll"
#define __SYCL_CUDA_PLUGIN_NAME "pi_cuda.dll"
#define __SYCL_ESIMD_EMULATOR_PLUGIN_NAME "pi_esimd_emulator.dll"
#define __SYCL_HIP_PLUGIN_NAME "pi_hip.dll"
#define __SYCL_UNIFIED_RUNTIME_PLUGIN_NAME "pi_unified_runtime.dll"
#define __SYCL_NATIVE_CPU_PLUGIN_NAME "pi_native_cpu.dll"
#else // llvm-mingw
#define __SYCL_OPENCL_PLUGIN_NAME "libpi_opencl.dll"
#define __SYCL_LEVEL_ZERO_PLUGIN_NAME "libpi_level_zero.dll"
#define __SYCL_CUDA_PLUGIN_NAME "libpi_cuda.dll"
#define __SYCL_ESIMD_EMULATOR_PLUGIN_NAME "libpi_esimd_emulator.dll"
#define __SYCL_HIP_PLUGIN_NAME "libpi_hip.dll"
#define __SYCL_UNIFIED_RUNTIME_PLUGIN_NAME "libpi_unified_runtime.dll"
#define __SYCL_NATIVE_CPU_PLUGIN_NAME "libpi_native_cpu.dll"
#endif

// ------------------------------------

using MapT = std::map<std::filesystem::path, void *>;

MapT &getDllMap() {
  static MapT dllMap;
  return dllMap;
}

/// Load the plugin libraries and store them in a map.
void preloadLibraries() {
  // Suppress system errors.
  // Tells the system to not display the critical-error-handler message box.
  // Instead, the system sends the error to the calling process.
  // This is crucial for graceful handling of plugins that couldn't be
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

  // this path duplicates sycl/detail/pi.cpp:initializePlugins
  std::filesystem::path LibSYCLDir(getCurrentDSODir());

  MapT &dllMap = getDllMap();

  // When searching for dependencies of the plugins limit the
  // list of directories to %windows%\system32 and the directory that contains
  // the loaded DLL (the plugin). This is necessary to avoid loading dlls from
  // current directory and some other directories which are considered unsafe.
  auto loadPlugin = [&](auto pluginName,
                        DWORD flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                      LOAD_LIBRARY_SEARCH_SYSTEM32) {
    auto path = LibSYCLDir / pluginName;
    dllMap.emplace(path, LoadLibraryEx(path.wstring().c_str(), NULL, flags));
  };
  loadPlugin(__SYCL_OPENCL_PLUGIN_NAME);
  loadPlugin(__SYCL_LEVEL_ZERO_PLUGIN_NAME);
  loadPlugin(__SYCL_CUDA_PLUGIN_NAME);
  loadPlugin(__SYCL_ESIMD_EMULATOR_PLUGIN_NAME);
  loadPlugin(__SYCL_HIP_PLUGIN_NAME);
  loadPlugin(__SYCL_UNIFIED_RUNTIME_PLUGIN_NAME);
  loadPlugin(__SYCL_NATIVE_CPU_PLUGIN_NAME);

  // Restore system error handling.
  (void)SetErrorMode(SavedMode);
  if (!SetDllDirectory(nullptr)) {
    assert(false && "Failed to restore DLL search path");
  }
}

/// windows_pi.cpp:loadOsPluginLibrary() calls this to get the DLL loaded
/// earlier.
__declspec(dllexport) void *getPreloadedPlugin(
    const std::filesystem::path &PluginPath) {

  MapT &dllMap = getDllMap();

  // All entries in the dllMap have the same parent directory.
  // To avoid case sensivity issues, we don't want to do string comparison but
  // just make sure that directory of the entires in the map and directory of
  // the PluginPath are equivalent (point to the same physical location).
  auto match = dllMap.end();
  std::error_code ec;
  if (!dllMap.empty() &&
      std::filesystem::equivalent((dllMap.begin())->first.parent_path(),
                                  PluginPath.parent_path(), ec)) {
    // Now we can search only by filename. Result might be nullptr (not found),
    // which is perfectly valid.
    match =
        std::find_if(dllMap.begin(), dllMap.end(),
                     [&](const std::pair<std::filesystem::path, void *> &v) {
                       return v.first.filename() == PluginPath.filename();
                     });
  }

  if (match == dllMap.end()) {
    // unit testing? return nullptr (not found) rather than risk asserting below
    if (PluginPath.string().find("unittests") != std::string::npos)
      return nullptr;

    // Otherwise, asking for something we don't know about at all, is an issue.
    std::cout << "unknown plugin: " << PluginPath << std::endl;
    assert(false && "getPreloadedPlugin was given an unknown plugin path.");
    return nullptr;
  }
  return match->second;
}

__declspec(dllexport) void *getPreloadedPlugin(const std::string &PluginPath) {
  return getPreloadedPlugin(std::filesystem::path(PluginPath));
}

BOOL WINAPI DllMain(HINSTANCE hinstDLL, // handle to DLL module
                    DWORD fdwReason,    // reason for calling function
                    LPVOID lpReserved)  // reserved
{
  bool PrintPiTrace = false;
  static const char *PiTrace = std::getenv("SYCL_PI_TRACE");
  static const int PiTraceValue = PiTrace ? std::stoi(PiTrace) : 0;
  if (PiTraceValue == -1 || PiTraceValue == 2) { // Means print all PI traces
    PrintPiTrace = true;
  }

  switch (fdwReason) {
  case DLL_PROCESS_ATTACH:
    if (PrintPiTrace)
      std::cout << "---> DLL_PROCESS_ATTACH pi_win_proxy_loader.dll\n"
                << std::endl;

    preloadLibraries();
    break;
  case DLL_PROCESS_DETACH:
    if (PrintPiTrace)
      std::cout << "---> DLL_PROCESS_DETACH pi_win_proxy_loader.dll\n"
                << std::endl;

  case DLL_THREAD_ATTACH:
  case DLL_THREAD_DETACH:
    break;
  }
  return TRUE;
}

#endif // WIN32
