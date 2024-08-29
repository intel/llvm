//===--------- context.hpp - Level Zero Adapter --------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <ur_api.h>

#include "command_list_cache.hpp"
#include "event_pool_cache.hpp"

struct ur_context_handle_t_ : _ur_object {
  ur_context_handle_t_(ze_context_handle_t hContext, uint32_t numDevices,
                       const ur_device_handle_t *phDevices, bool ownZeContext);
  ~ur_context_handle_t_() noexcept(false);

  ur_result_t retain();
  ur_result_t release();

  ze_context_handle_t getZeHandle() const;
  ur_platform_handle_t getPlatform() const;
  const std::vector<ur_device_handle_t> &getDevices() const;

  // Checks if Device is covered by this context.
  // For that the Device or its root devices need to be in the context.
  bool isValidDevice(ur_device_handle_t Device) const;

  const ze_context_handle_t hContext;
  const std::vector<ur_device_handle_t> hDevices;
  v2::command_list_cache_t commandListCache;
  v2::event_pool_cache eventPoolCache;
};
