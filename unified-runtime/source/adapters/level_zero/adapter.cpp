//===--------- adapter.cpp - Level Zero V1 Adapter -----------------------===//
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

#include "common.hpp"
#include "common/adapter.hpp"
#include "common/device.hpp"
#include "common/platform.hpp"
#include "ur_interface_loader.hpp"

namespace ur::level_zero::v1 {

ur_result_t urAdapterGet(
    /// [in] the number of platforms to be added to phAdapters. If phAdapters is
    /// not NULL, then NumEntries should be greater than zero, otherwise
    /// ::UR_RESULT_ERROR_INVALID_SIZE, will be returned.
    uint32_t NumEntries,
    /// [out][optional][range(0, NumEntries)] array of handle of adapters.
    /// If NumEntries is less than the number of adapters available, then
    /// ::urAdapterGet shall only retrieve that number of platforms.
    ::ur_adapter_handle_t *AdaptersOpque,
    /// [out][optional] returns the total number of adapters available.
    uint32_t *NumAdapters) try {
  std::lock_guard<std::mutex> Lock(GlobalAdapterMutex);

  // L0v2 adapter may have already been selected. Return adapter only if it
  // carries our DDI. If the GlobalAdapter is not set, then create a new one
  // and return it.
  if (GlobalAdapter) {
    if (GlobalAdapter->ddi_table != ddi_getter::value()) {
      if (NumAdapters)
        *NumAdapters = 0;
      return UR_RESULT_SUCCESS;
    }
    if (NumEntries && AdaptersOpque) {
      *AdaptersOpque = common_cast(GlobalAdapter);
      GlobalAdapter->RefCount.retain();
    }
    if (NumAdapters)
      *NumAdapters = 1;
    return UR_RESULT_SUCCESS;
  }

  auto *adapter = new ur_adapter_handle_t_();
  if (adapter->Platforms.empty()) {
    delete adapter;
    releaseStaticLoaderResourcesIfNeeded();

    if (NumAdapters)
      *NumAdapters = 0;
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
  }

  if (!ur_getenv("UR_ADAPTERS_FORCE_LOAD").has_value()) {
    auto [useV1, reason] = shouldUseV1Adapter();
    if (!useV1) {
      UR_LOG(INFO, "Skipping L0 V1 adapter: {}", reason);
      delete adapter;
#ifndef UR_STATIC_ADAPTER_LEVEL_ZERO
      releaseStaticLoaderResourcesIfNeeded();
#endif
      if (NumAdapters)
        *NumAdapters = 0;
      return UR_RESULT_SUCCESS;
    }
  }

  // Stamp L0v1 DDI table on the created adapter and its platforms.
  adapter->ddi_table = ddi_getter::value();
  for (auto &p : adapter->Platforms)
    p->ddi_table = adapter->ddi_table;
  GlobalAdapter = adapter;

#if defined(_WIN32)
  // On Windows the adapter library is torn down before the UR loader,
  // so register an atexit cleanup that frees GlobalAdapter ourselves.
  std::atexit(ur::level_zero::globalAdapterOnDemandCleanup);
#endif

  if (NumEntries && AdaptersOpque) {
    *AdaptersOpque = common_cast(GlobalAdapter);
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

} // namespace ur::level_zero::v1
