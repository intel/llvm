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

struct ur_context_handle_t_ : _ur_object {
  ur_context_handle_t_(ze_context_handle_t hContext, uint32_t numDevices,
                       const ur_device_handle_t *phDevices, bool ownZeContext);
  ~ur_context_handle_t_() noexcept(false);

  ur_result_t retain();
  ur_result_t release();

  const ze_context_handle_t hContext;
  const std::vector<ur_device_handle_t> hDevices;
  v2::command_list_cache_t commandListCache;
};
