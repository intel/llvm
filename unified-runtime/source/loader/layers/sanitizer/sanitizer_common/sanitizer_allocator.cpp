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
void validateDeviceUSM(uptr Allocated, DeviceType DeviceType) {
  switch (DeviceType) {
  case DeviceType::GPU_PVC: {
    assert((Allocated >> 52) == 0xff0);
    break;
  default:
    break;
  }
  }
}

void validateSharedUSM(uptr Allocated, DeviceType DeviceType) {
  switch (DeviceType) {
  case DeviceType::GPU_PVC: {
    assert((Allocated >> 40) == 0x7f);
    break;
  default:
    break;
  }
  }
}
} // namespace

ur_result_t SafeAllocate(ur_context_handle_t Context, ur_device_handle_t Device,
                         uptr Size, const ur_usm_desc_t *Properties,
                         ur_usm_pool_handle_t Pool, AllocType Type,
                         void **Allocated) {
  DeviceType DevieType =
      Device ? GetDeviceType(Context, Device) : DeviceType::UNKNOWN;
  switch (Type) {
  case AllocType::DEVICE_USM:
  case AllocType::MEM_BUFFER:
    UR_CALL(getContext()->urDdiTable.USM.pfnDeviceAlloc(
        Context, Device, Properties, Pool, Size, Allocated));
    validateDeviceUSM((uptr)*Allocated, DevieType);
    break;
  case AllocType::HOST_USM:
    UR_CALL(getContext()->urDdiTable.USM.pfnHostAlloc(Context, Properties, Pool,
                                                      Size, Allocated));
    // FIXME: it's hard to validate host USM pointer because we don't have
    // device information here
    break;
  case AllocType::SHARED_USM:
    UR_CALL(getContext()->urDdiTable.USM.pfnSharedAlloc(
        Context, Device, Properties, Pool, Size, Allocated));
    validateSharedUSM((uptr)*Allocated, DevieType);
    break;
  default:
    UR_LOG_L(getContext()->logger, ERR, "Unsupport memory type: {}",
             ToString(Type));
    return UR_RESULT_ERROR_INVALID_ARGUMENT;
  }
  return UR_RESULT_SUCCESS;
}

} // namespace ur_sanitizer_layer
