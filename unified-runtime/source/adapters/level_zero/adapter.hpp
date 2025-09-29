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

struct ur_adapter_handle_t_ : ur::handle_base<ur::level_zero::ddi_getter> {
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

  ur::RefCount RefCount;
};

extern ur_adapter_handle_t_ *GlobalAdapter;
