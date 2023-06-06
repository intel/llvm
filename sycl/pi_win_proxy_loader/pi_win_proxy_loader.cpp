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

static constexpr const char *DirSep = "\\";
using OSModuleHandle = intptr_t;
/// Module handle for the executable module - it is assumed there is always
/// single one at most.
static constexpr OSModuleHandle ExeModuleHandle = -1;

// cribbed from sycl/source/detail/os_util.cpp
std::string getDirName(const char *Path) {
  std::string Tmp(Path);
  // Remove trailing directory separators
  Tmp.erase(Tmp.find_last_not_of("/\\") + 1, std::string::npos);

  size_t pos = Tmp.find_last_of("/\\");
  if (pos != std::string::npos)
    return Tmp.substr(0, pos);

  // If no directory separator is present return initial path like dirname does
  return Tmp;
}

// cribbed from sycl/source/detail/os_util.cpp
OSModuleHandle getOSModuleHandle(const void *VirtAddr) {
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
std::string getCurrentDSODir() {
  char Path[MAX_PATH];
  Path[0] = '\0';
  Path[sizeof(Path) - 1] = '\0';
  auto Handle = getOSModuleHandle(reinterpret_cast<void *>(&getCurrentDSODir));
  DWORD Ret = GetModuleFileNameA(
      reinterpret_cast<HMODULE>(ExeModuleHandle == Handle ? 0 : Handle),
      reinterpret_cast<LPSTR>(&Path), sizeof(Path));
  assert(Ret < sizeof(Path) && "Path is longer than PATH_MAX?");
  assert(Ret > 0 && "GetModuleFileNameA failed");
  (void)Ret;

  BOOL RetCode = PathRemoveFileSpecA(reinterpret_cast<LPSTR>(&Path));
  assert(RetCode && "PathRemoveFileSpecA failed");
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
#define __SYCL_HIP_PLUGIN_NAME "libpi_hip.dll"
#define __SYCL_UNIFIED_RUNTIME_PLUGIN_NAME "pi_unified_runtime.dll"
#define __SYCL_ONLINE_COMPILER_LIBRARY_NAME "ocloc64.dll"
#else // llvm-mingw
#define __SYCL_OPENCL_PLUGIN_NAME "libpi_opencl.dll"
#define __SYCL_LEVEL_ZERO_PLUGIN_NAME "libpi_level_zero.dll"
#define __SYCL_CUDA_PLUGIN_NAME "libpi_cuda.dll"
#define __SYCL_ESIMD_EMULATOR_PLUGIN_NAME "libpi_esimd_emulator.dll"
#define __SYCL_HIP_PLUGIN_NAME "libpi_hip.dll"
#define __SYCL_UNIFIED_RUNTIME_PLUGIN_NAME "libpi_unified_runtime.dll"
#define __SYCL_ONLINE_COMPILER_LIBRARY_NAME "ocloc64.dll"
#endif

// ------------------------------------

using MapT = std::map<std::string, void *>;

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
  if (!SetDllDirectoryA("")) {
    assert(false && "Failed to update DLL search path");
  }

  // this path duplicates sycl/detail/pi.cpp:initializePlugins
  const std::string LibSYCLDir = getCurrentDSODir() + DirSep;

  MapT &dllMap = getDllMap();

  std::string ocl_path = LibSYCLDir + __SYCL_OPENCL_PLUGIN_NAME;
  dllMap.emplace(ocl_path, LoadLibraryA(ocl_path.c_str()));

  std::string l0_path = LibSYCLDir + __SYCL_LEVEL_ZERO_PLUGIN_NAME;
  dllMap.emplace(l0_path, LoadLibraryA(l0_path.c_str()));

  std::string cuda_path = LibSYCLDir + __SYCL_CUDA_PLUGIN_NAME;
  dllMap.emplace(cuda_path, LoadLibraryA(cuda_path.c_str()));

  std::string esimd_path = LibSYCLDir + __SYCL_ESIMD_EMULATOR_PLUGIN_NAME;
  dllMap.emplace(esimd_path, LoadLibraryA(esimd_path.c_str()));

  std::string hip_path = LibSYCLDir + __SYCL_HIP_PLUGIN_NAME;
  dllMap.emplace(hip_path, LoadLibraryA(hip_path.c_str()));

  std::string ur_path = LibSYCLDir + __SYCL_UNIFIED_RUNTIME_PLUGIN_NAME;
  dllMap.emplace(ur_path, LoadLibraryA(ur_path.c_str()));

  std::string ocloc_path = __SYCL_ONLINE_COMPILER_LIBRARY_NAME;
  dllMap.emplace(ocloc_path, LoadLibraryA(ocloc_path.c_str()));

  // Restore system error handling.
  (void)SetErrorMode(SavedMode);
  if (!SetDllDirectoryA(nullptr)) {
    assert(false && "Failed to restore DLL search path");
  }
}

/// windows_pi.cpp:loadOsPluginLibrary() calls this to get the DLL loaded
/// earlier.
__declspec(dllexport) void *getPreloadedPlugin(const std::string &PluginPath) {

  MapT &dllMap = getDllMap();

  auto match = dllMap.find(PluginPath); // result might be nullptr (not found),
                                        // which is perfectly valid.
  if (match == dllMap.end()) {
    // unit testing? return nullptr (not found) rather than risk asserting below
    if (PluginPath.find("unittests") != std::string::npos)
      return nullptr;

    // Otherwise, asking for something we don't know about at all, is an issue.
    std::cout << "unknown plugin: " << PluginPath << std::endl;
    assert(false && "getPreloadedPlugin was given an unknown plugin path.");
    return nullptr;
  }
  return match->second;
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
