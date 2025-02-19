// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UUR_USM_HELPERS_H_INCLUDED
#define UUR_USM_HELPERS_H_INCLUDED

#include <uur/fixtures.h>

namespace uur {

using USMAllocTestParams =
    std::tuple<uur::BoolTestParam, uint32_t, size_t, ur_usm_advice_flag_t>;

struct urUSMAllocTest : uur::urQueueTestWithParam<uur::USMAllocTestParams> {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::urQueueTestWithParam<uur::USMAllocTestParams>::SetUp());
    if (usePool) {
      ur_bool_t poolSupport = false;
      ASSERT_SUCCESS(uur::GetDeviceUSMPoolSupport(device, poolSupport));
      if (!poolSupport) {
        GTEST_SKIP() << "USM pools are not supported.";
      }
      ur_usm_pool_desc_t pool_desc = {};
      ASSERT_SUCCESS(urUSMPoolCreate(context, &pool_desc, &pool));
    }
  }

  void TearDown() override {
    if (pool) {
      ASSERT_SUCCESS(urUSMPoolRelease(pool));
    }
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::urQueueTestWithParam<uur::USMAllocTestParams>::TearDown());
  }

  ur_usm_pool_handle_t pool = nullptr;
  const bool usePool = std::get<0>(getParam()).value;
  ur_device_usm_access_capability_flags_t USMSupport = 0;
  const uint32_t alignment = std::get<1>(getParam());
  size_t allocation_size = std::get<2>(getParam());
  const ur_usm_advice_flag_t advice_flags = std::get<3>(getParam());
  void *ptr = nullptr;
};

template <typename T>
inline std::string printUSMAllocTestString(
    const testing::TestParamInfo<typename T::ParamType> &info) {
  // ParamType will be std::tuple<ur_device_handle_t, USMAllocTestParams>
  const auto device_handle = std::get<0>(info.param).device;
  const auto platform_device_name =
      uur::GetPlatformAndDeviceName(device_handle);
  const auto &USMAllocTestParams = std::get<1>(info.param);
  const auto &BoolParam = std::get<0>(USMAllocTestParams);

  std::stringstream ss;
  ss << BoolParam.name << (BoolParam.value ? "Enabled" : "Disabled"); // UsePool
  ss << "_";
  ss << std::get<1>(USMAllocTestParams); // alignment
  ss << "_";
  ss << std::get<2>(USMAllocTestParams); // size
  ss << "_";
  ss << std::get<3>(USMAllocTestParams); // ur_usm_advice_flags_t

  return platform_device_name + "__" + ss.str();
}

static std::vector<ur_usm_advice_flag_t> usm_advice_test_parameters{
    UR_USM_ADVICE_FLAG_DEFAULT,
    UR_USM_ADVICE_FLAG_SET_READ_MOSTLY,
    UR_USM_ADVICE_FLAG_CLEAR_READ_MOSTLY,
    UR_USM_ADVICE_FLAG_SET_PREFERRED_LOCATION,
    UR_USM_ADVICE_FLAG_CLEAR_PREFERRED_LOCATION,
    UR_USM_ADVICE_FLAG_SET_NON_ATOMIC_MOSTLY,
    UR_USM_ADVICE_FLAG_CLEAR_NON_ATOMIC_MOSTLY,
    UR_USM_ADVICE_FLAG_BIAS_CACHED,
    UR_USM_ADVICE_FLAG_BIAS_UNCACHED,
    UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_DEVICE,
    UR_USM_ADVICE_FLAG_CLEAR_ACCESSED_BY_DEVICE,
    UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_HOST,
    UR_USM_ADVICE_FLAG_CLEAR_ACCESSED_BY_HOST,
    UR_USM_ADVICE_FLAG_SET_PREFERRED_LOCATION_HOST,
    UR_USM_ADVICE_FLAG_CLEAR_PREFERRED_LOCATION_HOST,
    UR_USM_ADVICE_FLAG_SET_NON_COHERENT_MEMORY,
    UR_USM_ADVICE_FLAG_CLEAR_NON_COHERENT_MEMORY};

} // namespace uur

#endif // UUR_ENQUEUE_RECT_HELPERS_H_INCLUDED
