/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file sanitizer_allocator.cpp
 *
 */

#include "sanitizer_allocator.hpp"
#include "sanitizer_libdevice.hpp"
#include "sanitizer_utils.hpp"
#include "ur_sanitizer_layer.hpp"

namespace ur_sanitizer_layer {

namespace {
void validate(uptr Allocated, AllocType AllocType, DeviceType DeviceType) {
  if (DeviceType == DeviceType::GPU_PVC) {
    switch (AllocType) {
    case AllocType::DEVICE_USM:
      assert((Allocated >> 52) == 0xff0);
      break;
    case AllocType::HOST_USM:
      assert((Allocated >> 40) == 0xffff);
      break;
    case AllocType::SHARED_USM:
      assert((Allocated >> 40) == 0x7f);
      break;
    default:
      return;
    }
  }
}
} // namespace

void *Allocator::allocate(uptr Size, const ur_usm_desc_t *Properties,
                          AllocType Type) {
  void *Allocated = nullptr;
  ur_result_t Result;
  ur_usm_pool_handle_t Pool = nullptr;

  if (Type == AllocType::DEVICE_USM) {
    Result = getContext()->urDdiTable.USM.pfnDeviceAlloc(
        Context, Device, Properties, Pool, Size, &Allocated);
  } else if (Type == AllocType::HOST_USM) {
    Result = getContext()->urDdiTable.USM.pfnHostAlloc(Context, Properties,
                                                       Pool, Size, &Allocated);
  } else if (Type == AllocType::SHARED_USM) {
    Result = getContext()->urDdiTable.USM.pfnSharedAlloc(
        Context, Device, Properties, Pool, Size, &Allocated);
  } else {
    return nullptr;
  }

  if (Result != UR_RESULT_SUCCESS) {
    return nullptr;
  }

  validate((uptr)Allocated, Type, GetDeviceType(Context, Device));

  return Allocated;
}

} // namespace ur_sanitizer_layer
