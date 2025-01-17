// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#ifndef UUR_USM_HELPERS_H_INCLUDED
#define UUR_USM_HELPERS_H_INCLUDED

#include <uur/fixtures.h>

namespace uur {

using USMDeviceAllocParams = std::tuple<uur::BoolTestParam, uint32_t, size_t>;

template <typename T>
inline std::string printUSMAllocTestString(
    const testing::TestParamInfo<typename T::ParamType> &info) {
  // ParamType will be std::tuple<ur_device_handle_t, USMDeviceAllocParams>
  const auto device_handle = std::get<0>(info.param).device;
  const auto platform_device_name =
      uur::GetPlatformAndDeviceName(device_handle);
  const auto &usmDeviceAllocParams = std::get<1>(info.param);
  const auto &BoolParam = std::get<0>(usmDeviceAllocParams);

  std::stringstream ss;
  ss << BoolParam.name << (BoolParam.value ? "Enabled" : "Disabled");

  const auto alignment = std::get<1>(usmDeviceAllocParams);
  const auto size = std::get<2>(usmDeviceAllocParams);
  if (alignment && size > 0) {
    ss << "_";
    ss << std::get<1>(usmDeviceAllocParams);
    ss << "_";
    ss << std::get<2>(usmDeviceAllocParams);
  }

  return platform_device_name + "__" + ss.str();
}

} // namespace uur

#endif // UUR_ENQUEUE_RECT_HELPERS_H_INCLUDED
