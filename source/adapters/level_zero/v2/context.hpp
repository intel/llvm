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

#include "command_list_cache.hpp"

#include "../context.hpp"

namespace v2 {

struct ur_context_handle_t_;
using ur_context_handle_t = ur_context_handle_t_ *;

struct ur_context_handle_t_ : public ::ur_context_handle_t_ {
  ur_context_handle_t_(ze_context_handle_t hContext, uint32_t numDevices,
                       const ur_device_handle_t *phDevices, bool ownZeContext);

  command_list_cache_t commandListCache;
};

} // namespace v2
