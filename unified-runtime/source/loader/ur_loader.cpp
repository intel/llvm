/*
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM
 * Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#include "ur_loader.hpp"
#ifdef UR_STATIC_ADAPTER_LEVEL_ZERO
#include "adapters/level_zero/ur_interface_loader.hpp"
#endif
#ifdef UR_STATIC_ADAPTER_OPENCL
#include "adapters/opencl/ur_interface_loader.hpp"
#endif

namespace ur_loader {
///////////////////////////////////////////////////////////////////////////////
context_t *getContext() { return context_t::get_direct(); }

ur_result_t context_t::init() {
#ifdef _WIN32
  // Suppress system errors.
  // Tells the system to not display the critical-error-handler message box.
  // Instead, the system sends the error to the calling process.
  // This is crucial for graceful handling of adapters that couldn't be
  // loaded, e.g. due to missing native run-times.
  // TODO: add reporting in case of an error.
  // NOTE: we restore the old mode to not affect user app behavior.
  // See
  // https://github.com/intel/llvm/blob/sycl/sycl/ur_win_proxy_loader/ur_win_proxy_loader.cpp
  // (preloadLibraries())
  UINT SavedMode = SetErrorMode(SEM_FAILCRITICALERRORS);
#endif

#if defined(UR_STATIC_ADAPTER_LEVEL_ZERO) || defined(UR_STATIC_ADAPTER_OPENCL)
  // If the adapters were force loaded, it means the user wants to use
  // a specific adapter library. Don't load any static adapters.
  [[maybe_unused]] bool staticOCLRegistered = false;
  if (!adapter_registry.adaptersForceLoaded()) {
#ifdef UR_STATIC_ADAPTER_LEVEL_ZERO
    auto &level_zero = platforms.emplace_back(nullptr);
    ur::level_zero::urAdapterGetDdiTables(&level_zero.dditable);
#endif
#ifdef UR_STATIC_ADAPTER_OPENCL
    {
      // Probe the adapter before registering: urAdapterGet triggers
      // loadOCLLibrary(). If the OpenCL library is absent the global
      // ocl::*_ptr function pointers stay null, and any later call through
      // the DDI table (e.g. urPlatformGetInfo) would segfault.
      ur_dditable_t ocl_ddi = {};
      ur::opencl::urAdapterGetDdiTables(&ocl_ddi);
      ur_adapter_handle_t hAdapter = nullptr;
      if (ocl_ddi.Adapter.pfnGet &&
          ocl_ddi.Adapter.pfnGet(1, &hAdapter, nullptr) == UR_RESULT_SUCCESS) {
        ocl_ddi.Adapter.pfnRelease(hAdapter);
        auto &opencl = platforms.emplace_back(nullptr);
        opencl.dditable = ocl_ddi;
        staticOCLRegistered = true;
      }
    }
#endif
  }
#endif

  for (const auto &adapterPaths : adapter_registry) {
#ifdef UR_STATIC_ADAPTER_OPENCL
    // Skip the dynamic OpenCL adapter when the static one is already
    // registered — loading both would double-register the OCL backend.
    if (staticOCLRegistered && !adapterPaths.empty() &&
        adapterPaths[0].string().find("ur_adapter_opencl") != std::string::npos)
      continue;
#endif
    for (const auto &path : adapterPaths) {
      auto handle = LibLoader::loadAdapterLibrary(path.string().c_str());
      if (handle) {
        platforms.emplace_back(std::move(handle));
        break;
      }
    }
  }
#ifdef _WIN32
  // Restore system error handling.
  (void)SetErrorMode(SavedMode);
#endif

  forceIntercept = getenv_tobool("UR_ENABLE_LOADER_INTERCEPT");

  if (forceIntercept || platforms.size() > 1) {
    intercept_enabled = true;
  }

  return UR_RESULT_SUCCESS;
}

} // namespace ur_loader
