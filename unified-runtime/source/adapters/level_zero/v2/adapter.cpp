//===--------- adapter.cpp - Level Zero Adapter v2 -----------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdlib>
#include <mutex>
#include <unified-runtime/ur_api.h>
#include <ur/ur.hpp>

#include "../common/adapter.hpp"
#include "../common/device.hpp"
#include "../common/platform.hpp"
#include "ur_interface_loader.hpp"

namespace ur::level_zero::v2 {

ur_result_t urAdapterGet(uint32_t NumEntries, ::ur_adapter_handle_t *Adapters,
                         uint32_t *NumAdapters) try {
  std::lock_guard<std::mutex> Lock(GlobalAdapterMutex);

  // L0v1 adapter may have already been selected. Return adapter only if it
  // carries our DDI. If the GlobalAdapter is not set, then create a new one
  // and return it.
  if (GlobalAdapter) {
    if (GlobalAdapter->ddi_table != ur::level_zero::v2::ddi_getter::value()) {
      if (NumAdapters)
        *NumAdapters = 0;
      return UR_RESULT_SUCCESS;
    }
    if (NumEntries && Adapters) {
      *Adapters = common_cast(GlobalAdapter);
      GlobalAdapter->RefCount.retain();
    }
    if (NumAdapters)
      *NumAdapters = 1;
    return UR_RESULT_SUCCESS;
  }

  auto *adapter = new ur::level_zero::ur_adapter_handle_t_();
  if (adapter->Platforms.empty()) {
    delete adapter;

    releaseStaticLoaderResourcesIfNeeded();
    if (NumAdapters)
      *NumAdapters = 0;
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
  }
  adapter->version = 2;

  if (!ur_getenv("UR_ADAPTERS_FORCE_LOAD").has_value()) {
    auto [useV2, reason] = shouldUseV2Adapter();
    if (!useV2) {
      UR_LOG(INFO, "Skipping L0 V2 adapter: {}", reason);
      delete adapter;
#ifndef UR_STATIC_ADAPTER_LEVEL_ZERO_V2
      releaseStaticLoaderResourcesIfNeeded();
#endif
      if (NumAdapters)
        *NumAdapters = 0;
      return UR_RESULT_SUCCESS;
    }
  }

  // Stamp L0v2 DDI table on the created adapter and its platforms.
  adapter->ddi_table = ddi_getter::value();
  for (auto &platform : adapter->Platforms)
    platform->ddi_table = adapter->ddi_table;
  GlobalAdapter = adapter;

#if defined(_WIN32)
  // On Windows the adapter library is torn down before the UR loader,
  // so register an atexit cleanup that frees GlobalAdapter ourselves.
  std::atexit(ur::level_zero::globalAdapterOnDemandCleanup);
#endif

  if (NumEntries && Adapters) {
    *Adapters = common_cast(GlobalAdapter);
    if (GlobalAdapter->RefCount.retain() == 0) {
      adapterStateInit();
    }
  }

  if (NumAdapters)
    *NumAdapters = 1;

  return UR_RESULT_SUCCESS;
} catch (ur_result_t result) {
  return result;
} catch (...) {
  return UR_RESULT_ERROR_UNKNOWN;
}

} // namespace ur::level_zero::v2
