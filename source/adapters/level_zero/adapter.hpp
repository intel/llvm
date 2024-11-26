//===--------- adapters.hpp - Level Zero Adapter --------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "logger/ur_logger.hpp"
#include <atomic>
#include <loader/ur_loader.hpp>
#include <loader/ze_loader.h>
#include <mutex>
#include <optional>
#include <ur/ur.hpp>
#include <ze_api.h>
#include <ze_ddi.h>
#include <zes_ddi.h>

using PlatformVec = std::vector<std::unique_ptr<ur_platform_handle_t_>>;

class ur_legacy_sink;

struct ur_adapter_handle_t_ {
  ur_adapter_handle_t_();
  std::atomic<uint32_t> RefCount = 0;
  std::mutex Mutex;

  zes_pfnDriverGetDeviceByUuidExp_t getDeviceByUUIdFunctionPtr = nullptr;
  zes_pfnDriverGet_t getSysManDriversFunctionPtr = nullptr;
  zes_pfnInit_t sysManInitFunctionPtr = nullptr;
  ze_pfnInitDrivers_t initDriversFunctionPtr = nullptr;
  ze_init_driver_type_desc_t InitDriversDesc = {
      ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC, nullptr,
      ZE_INIT_DRIVER_TYPE_FLAG_GPU};
  uint32_t ZeInitDriversCount = 0;
  bool InitDriversSupported = false;

  std::optional<ze_result_t> ZeInitDriversResult;
  std::optional<ze_result_t> ZeInitResult;
  std::optional<ze_result_t> ZeResult;
  std::optional<ze_result_t> ZesResult;
  ZeCache<Result<PlatformVec>> PlatformCache;
  logger::Logger &logger;
  HMODULE processHandle = nullptr;
};

extern ur_adapter_handle_t_ *GlobalAdapter;
