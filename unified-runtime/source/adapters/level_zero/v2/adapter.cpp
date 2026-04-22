//===--------- adapter.cpp - Level Zero Adapter v2 -----------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mutex>
#include <ur/ur.hpp>
#include <ur_api.h>

#include "../adapter.hpp"
#include "../device.hpp"
#include "../platform.hpp"
#include "ur_interface_loader.hpp"

ur_adapter_handle_t_ *GlobalAdapterV2 = nullptr;

namespace ur::level_zero_v2 {

ur_result_t urAdapterGetInfo(ur_adapter_handle_t hAdapter,
                             ur_adapter_info_t propName, size_t propSize,
                             void *propValue, size_t *propSizeRet) {
  if (propName == UR_ADAPTER_INFO_VERSION) {
    UrReturnHelper ReturnValue(propSize, propValue, propSizeRet);
    return ReturnValue(uint32_t{2});
  }
  return ur::level_zero::urAdapterGetInfo(hAdapter, propName, propSize,
                                          propValue, propSizeRet);
}

ur_result_t urAdapterGet(uint32_t NumEntries, ur_adapter_handle_t *Adapters,
                         uint32_t *NumAdapters) try {
  static std::mutex Mutex;
  std::lock_guard<std::mutex> Lock(Mutex);

  if (!GlobalAdapterV2) {
    GlobalAdapterV2 = new ur_adapter_handle_t_();

    if (GlobalAdapterV2->Platforms.empty()) {
      delete GlobalAdapterV2;
      GlobalAdapterV2 = nullptr;
      if (NumAdapters)
        *NumAdapters = 0;
      return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    // Override DDI table in the adapter and all platform handles so the
    // intercept layer dispatches through v2's DDI table.
    auto *v2Ddi = ur::level_zero_v2::ddi_getter::value();
    GlobalAdapterV2->ddi_table = v2Ddi;
    for (auto &platform : GlobalAdapterV2->Platforms)
      platform->ddi_table = v2Ddi;
  }

#ifdef UR_L0_V1_ADAPTER_ENABLED
  {
    auto [useV2, reason] = shouldUseV2Adapter();
    if (!useV2) {
      UR_LOG(INFO, "Skipping L0 V2 adapter: {}", reason);
      if (NumAdapters)
        *NumAdapters = 0;
      return UR_RESULT_SUCCESS;
    }
  }
#endif

  if (NumEntries && Adapters) {
    *Adapters = GlobalAdapterV2;
    if (GlobalAdapterV2->RefCount.retain() == 0) {
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

} // namespace ur::level_zero_v2