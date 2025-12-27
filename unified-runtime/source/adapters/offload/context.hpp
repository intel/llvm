//===----------- context.hpp - LLVM Offload Adapter  ----------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "adapter.hpp"
#include "common.hpp"
#include "device.hpp"
#include <OffloadAPI.h>
#include <unordered_map>
#include <ur_api.h>

struct alloc_info_t {
  ol_alloc_type_t Type;
  size_t Size;
};

struct ur_context_handle_t_ : RefCounted {
  ur_context_handle_t_(ur_device_handle_t hDevice) : Device{hDevice} {
    urDeviceRetain(Device);
  }
  ~ur_context_handle_t_() { urDeviceRelease(Device); }

  ur_device_handle_t Device;

  ol_result_t getAllocType(const void *UsmPtr, ol_alloc_type_t &Type) {
    auto Err = olGetMemInfo(UsmPtr, OL_MEM_INFO_TYPE, sizeof(Type), &Type);
    if (Err && Err->Code == OL_ERRC_NOT_FOUND) {
      // Treat unknown allocations as host
      Type = OL_ALLOC_TYPE_HOST;
      return OL_SUCCESS;
    }
    return Err;
  }
};
