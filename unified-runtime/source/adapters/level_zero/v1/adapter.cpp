//===--------- adapter.cpp - Level Zero V1 Adapter -----------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mutex>
#include <ur/ur.hpp>
#include <unified-runtime/ur_api.h>

#include "adapter.hpp"
#include "../adapter.hpp"
#include "../common.hpp"
#include "../device.hpp"
#include "../platform.hpp"
#include "../ur_interface_loader.hpp"

// Strong definition of the V1 global adapter sentinel. The matching weak
// fallback in common/adapter.cpp keeps the symbol resolved in v2-only
// builds where this TU is not linked.
ur_adapter_handle_t_ *GlobalAdapter = nullptr;

namespace ur::level_zero::v1 {

ur_result_t urAdapterGet(
    /// [in] the number of platforms to be added to phAdapters. If phAdapters is
    /// not NULL, then NumEntries should be greater than zero, otherwise
    /// ::UR_RESULT_ERROR_INVALID_SIZE, will be returned.
    uint32_t NumEntries,
    /// [out][optional][range(0, NumEntries)] array of handle of adapters.
    /// If NumEntries is less than the number of adapters available, then
    /// ::urAdapterGet shall only retrieve that number of platforms.
    ur_adapter_handle_t *Adapters,
    /// [out][optional] returns the total number of adapters available.
    uint32_t *NumAdapters) try {
  static std::mutex AdapterConstructionMutex{};
  std::lock_guard<std::mutex> Lock{AdapterConstructionMutex};

  if (!GlobalAdapter) {
    GlobalAdapter = new ur_adapter_handle_t_();

    // Stamp v1's DDI table onto the adapter and platforms so the loader's
    // intercept layer dispatches them through v1's entrypoints.
    if (auto *v1Ddi = ur::level_zero::ddi_getter::value()) {
      GlobalAdapter->ddi_table = v1Ddi;
      for (auto &p : GlobalAdapter->Platforms)
        p->ddi_table = v1Ddi;
    }
  }

  if (GlobalAdapter->Platforms.size() == 0) {
    if (NumAdapters) {
      *NumAdapters = 0;
    }
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
  }

  // Decline if hardware/env preference says V1 is not the right choice
  // on this machine. Mirrors upstream's selection (which lives in the
  // ur_adapter_handle_t_ ctor — done after L0 init so that the IP
  // version probe inside shouldUseV1Adapter() can enumerate drivers).
  // Runs unconditionally except when UR_ADAPTERS_FORCE_LOAD overrides.
  if (!ur_getenv("UR_ADAPTERS_FORCE_LOAD").has_value()) {
    auto [useV1, reason] = shouldUseV1Adapter();
    if (!useV1) {
      UR_LOG(INFO, "Skipping L0 V1 adapter: {}", reason);
      if (NumAdapters)
        *NumAdapters = 0;
      return UR_RESULT_SUCCESS;
    }
  }

  if (NumEntries && Adapters) {
    *Adapters = GlobalAdapter;

    if (GlobalAdapter->RefCount.retain() == 0) {
      adapterStateInit();
    }
  }

  if (NumAdapters) {
    *NumAdapters = 1;
  }

  return UR_RESULT_SUCCESS;
} catch (ur_result_t result) {
  return result;
} catch (...) {
  return UR_RESULT_ERROR_UNKNOWN;
}

} // namespace ur::level_zero::v1
