//===--------- queue_factory.cpp - Level Zero Adapter --------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "../queue.hpp"
#include "context.hpp"

#include "queue_immediate_in_order.hpp"

namespace v2 {

inline bool shouldUseQueueV2(ur_device_handle_t hDevice,
                             ur_queue_flags_t flags) {
  std::ignore = hDevice;
  std::ignore = flags;

  const char *UrRet = std::getenv("UR_L0_USE_QUEUE_V2");
  return UrRet && std::stoi(UrRet);
}

inline ur_queue_handle_t createQueue(::ur_context_handle_t hContext,
                                     ur_device_handle_t hDevice,
                                     const ur_queue_properties_t *pProps) {
  if (!shouldUseQueueV2(hDevice, pProps ? pProps->flags : ur_queue_flags_t{})) {
    throw UR_RESULT_ERROR_INVALID_ARGUMENT;
  }
  // TODO: For now, always use immediate, in-order
  return new ur_queue_immediate_in_order_t(
      static_cast<v2::ur_context_handle_t>(hContext), hDevice, pProps);
}

} // namespace v2
