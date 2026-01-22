//===--------- adapters.hpp - Level Zero Adapter --------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "common.hpp"
#include "common/ur_ref_count.hpp"
#include "logger/ur_logger.hpp"
#include "ur_interface_loader.hpp"
#include <loader/ur_loader.hpp>
#include <loader/ze_loader.h>
#include <optional>
#include <ur/ur.hpp>
#include <ze_api.h>
#include <ze_ddi.h>
#include <zes_ddi.h>

using PlatformVec = std::vector<std::unique_ptr<ur_platform_handle_t_>>;

class ur_legacy_sink;

struct ur_adapter_handle_t_ : ur_shared_handle_base_t {
  ur_adapter_handle_t_();

  zes_pfnDriverGetDeviceByUuidExp_t getDeviceByUUIdFunctionPtr = nullptr;
  zes_pfnDriverGet_t getSysManDriversFunctionPtr = nullptr;
  zes_pfnInit_t sysManInitFunctionPtr = nullptr;
  ze_pfnInitDrivers_t initDriversFunctionPtr = nullptr;
  ze_init_driver_type_desc_t InitDriversDesc = {
      ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC, nullptr,
      ZE_INIT_DRIVER_TYPE_FLAG_GPU};
  uint32_t ZeInitDriversCount = 0;
  bool InitDriversSupported = false;

  PlatformVec Platforms;
  logger::Logger &logger;
  HMODULE processHandle = nullptr;

  // The adapter version returned via UR_ADAPTER_INFO_VERSION. Each
  // adapter overrides this in its urAdapterGet body after construction.
  uint32_t version = 1;

  ur::RefCount RefCount;
};

extern ur_adapter_handle_t_ *GlobalAdapter;
extern ur_adapter_handle_t_ *GlobalAdapterV2;

std::pair<bool, std::string> shouldUseV1Adapter();
std::pair<bool, std::string> shouldUseV2Adapter();

ur_result_t adapterStateInit();
ur_result_t initPlatforms(ur_adapter_handle_t_ *adapter, PlatformVec &platforms,
                          ze_result_t ZesResult) noexcept;
