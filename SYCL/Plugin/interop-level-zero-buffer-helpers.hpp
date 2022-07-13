#pragma once
#include "interop-level-zero-buffer-helpers.hpp"
#include <sycl/sycl.hpp>
// clang-format off
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
// clang-format on

using namespace sycl;

bool is_discrete(const device &Device) {
  auto ZeDevice = get_native<backend::ext_oneapi_level_zero>(Device);
  ze_device_properties_t ZeDeviceProps;
  ZeDeviceProps.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  ZeDeviceProps.pNext = nullptr;
  zeDeviceGetProperties(ZeDevice, &ZeDeviceProps);
  return !(ZeDeviceProps.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED);
}
