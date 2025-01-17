/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
 * LLVM-exception
 *
 */
#include "ur_loader.hpp"
#ifdef UR_STATIC_ADAPTER_LEVEL_ZERO
#include "adapters/level_zero/ur_interface_loader.hpp"
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

#ifdef UR_STATIC_ADAPTER_LEVEL_ZERO
  // If the adapters were force loaded, it means the user wants to use
  // a specific adapter library. Don't load any static adapters.
  if (!adapter_registry.adaptersForceLoaded()) {
    auto &level_zero = platforms.emplace_back(nullptr);
    ur::level_zero::urAdapterGetDdiTables(&level_zero.dditable.ur);
  }
#endif

  for (const auto &adapterPaths : adapter_registry) {
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
