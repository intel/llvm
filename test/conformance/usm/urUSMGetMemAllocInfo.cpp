// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include <uur/fixtures.h>
#include <uur/known_failure.h>

struct urUSMGetMemAllocInfoPoolTest
    : uur::urUSMDeviceAllocTestWithParam<ur_usm_alloc_info_t> {
  void SetUp() override {
    // The setup for the parent fixture does a urQueueFlush, which isn't
    // supported by native cpu.
    UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

    use_pool = getParam() == UR_USM_ALLOC_INFO_POOL;
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::urUSMDeviceAllocTestWithParam<ur_usm_alloc_info_t>::SetUp());
  }
};

UUR_DEVICE_TEST_SUITE_P(urUSMGetMemAllocInfoPoolTest,
                        ::testing::Values(UR_USM_ALLOC_INFO_POOL),
                        uur::deviceTestWithParamPrinter<ur_usm_alloc_info_t>);

TEST_P(urUSMGetMemAllocInfoPoolTest, SuccessPool) {
  UUR_KNOWN_FAILURE_ON(uur::OpenCL{}, uur::LevelZeroV2{});

  size_t property_size = 0;
  ur_usm_alloc_info_t property_name = UR_USM_ALLOC_INFO_POOL;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urUSMGetMemAllocInfo(context, ptr, property_name, 0, nullptr,
                           &property_size),
      property_name);
  ASSERT_EQ(sizeof(ur_usm_pool_handle_t), property_size);

  ur_usm_pool_handle_t returned_pool = nullptr;
  ASSERT_SUCCESS(urUSMGetMemAllocInfo(context, ptr, property_name,
                                      property_size, &returned_pool, nullptr));

  ASSERT_EQ(returned_pool, pool);
}

using urUSMGetMemAllocInfoTest = uur::urUSMDeviceAllocTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urUSMGetMemAllocInfoTest);

TEST_P(urUSMGetMemAllocInfoTest, SuccessType) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  ur_usm_alloc_info_t property_name = UR_USM_ALLOC_INFO_TYPE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urUSMGetMemAllocInfo(context, ptr, property_name, 0, nullptr,
                           &property_size),
      property_name);
  ASSERT_EQ(sizeof(ur_usm_type_t), property_size);

  ur_usm_type_t returned_type = UR_USM_TYPE_FORCE_UINT32;
  ASSERT_SUCCESS(urUSMGetMemAllocInfo(context, ptr, property_name,
                                      property_size, &returned_type, nullptr));

  ASSERT_EQ(returned_type, UR_USM_TYPE_DEVICE);
}

TEST_P(urUSMGetMemAllocInfoTest, SuccessBasePtr) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  ur_usm_alloc_info_t property_name = UR_USM_ALLOC_INFO_BASE_PTR;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urUSMGetMemAllocInfo(context, ptr, property_name, 0, nullptr,
                           &property_size),
      property_name);
  ASSERT_GT(property_size, 0);

  void *returned_ptr = nullptr;
  ASSERT_SUCCESS(urUSMGetMemAllocInfo(context, ptr, property_name,
                                      property_size, &returned_ptr, nullptr));

  ASSERT_EQ(returned_ptr, ptr);
}

TEST_P(urUSMGetMemAllocInfoTest, SuccessSize) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  ur_usm_alloc_info_t property_name = UR_USM_ALLOC_INFO_SIZE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urUSMGetMemAllocInfo(context, ptr, property_name, 0, nullptr,
                           &property_size),
      property_name);
  ASSERT_EQ(sizeof(size_t), property_size);

  size_t returned_size = 0;
  ASSERT_SUCCESS(urUSMGetMemAllocInfo(context, ptr, property_name,
                                      property_size, &returned_size, nullptr));

  ASSERT_GE(returned_size, allocation_size);
}

TEST_P(urUSMGetMemAllocInfoTest, SuccessDevice) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  ur_usm_alloc_info_t property_name = UR_USM_ALLOC_INFO_DEVICE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urUSMGetMemAllocInfo(context, ptr, property_name, 0, nullptr,
                           &property_size),
      property_name);
  ASSERT_EQ(sizeof(ur_device_handle_t), property_size);

  ur_device_handle_t returned_device = nullptr;
  ASSERT_SUCCESS(urUSMGetMemAllocInfo(
      context, ptr, property_name, property_size, &returned_device, nullptr));

  ASSERT_EQ(returned_device, device);
}

TEST_P(urUSMGetMemAllocInfoTest, InvalidNullHandleContext) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  ur_usm_type_t USMType = UR_USM_TYPE_FORCE_UINT32;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_HANDLE,
      urUSMGetMemAllocInfo(nullptr, ptr, UR_USM_ALLOC_INFO_FORCE_UINT32,
                           sizeof(ur_usm_type_t), &USMType, nullptr));
}

TEST_P(urUSMGetMemAllocInfoTest, InvalidNullPointerMem) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  ur_usm_type_t USMType = UR_USM_TYPE_FORCE_UINT32;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_POINTER,
      urUSMGetMemAllocInfo(context, nullptr, UR_USM_ALLOC_INFO_FORCE_UINT32,
                           sizeof(ur_usm_type_t), &USMType, nullptr));
}

TEST_P(urUSMGetMemAllocInfoTest, InvalidEnumeration) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  ur_usm_type_t USMType = UR_USM_TYPE_FORCE_UINT32;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_ENUMERATION,
      urUSMGetMemAllocInfo(context, ptr, UR_USM_ALLOC_INFO_FORCE_UINT32,
                           sizeof(ur_usm_type_t), &USMType, nullptr));
}

TEST_P(urUSMGetMemAllocInfoTest, InvalidValuePropSize) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  ur_usm_type_t USMType = UR_USM_TYPE_FORCE_UINT32;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                   urUSMGetMemAllocInfo(context, ptr, UR_USM_ALLOC_INFO_TYPE,
                                        sizeof(ur_usm_type_t) - 1, &USMType,
                                        nullptr));
}
