// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/known_failure.h>

struct urKernelSetArgPointerTest : uur::urKernelExecutionTest {
  void SetUp() {
    program_name = "fill_usm";
    UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());
  }

  void TearDown() {
    if (allocation) {
      ASSERT_SUCCESS(urUSMFree(context, allocation));
    }
    UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::TearDown());
  }

  void ValidateAllocation(void *pointer) {
    for (size_t i = 0; i < array_size; i++) {
      ASSERT_EQ(static_cast<uint32_t *>(pointer)[i], data);
    }
  }

  void *allocation = nullptr;
  size_t array_size = 16;
  size_t allocation_size = array_size * sizeof(uint32_t);
  uint32_t data = 42;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelSetArgPointerTest);

TEST_P(urKernelSetArgPointerTest, SuccessHost) {
  ur_device_usm_access_capability_flags_t host_usm_flags = 0;
  ASSERT_SUCCESS(uur::GetDeviceUSMHostSupport(device, host_usm_flags));
  if (!(host_usm_flags & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
    GTEST_SKIP() << "Host USM is not supported.";
  }

  ASSERT_SUCCESS(
      urUSMHostAlloc(context, nullptr, nullptr, allocation_size, &allocation));
  ASSERT_NE(allocation, nullptr);

  ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, allocation));
  ASSERT_SUCCESS(urKernelSetArgValue(kernel, 1, sizeof(data), nullptr, &data));
  Launch1DRange(array_size);
  ValidateAllocation(allocation);
}

TEST_P(urKernelSetArgPointerTest, SuccessDevice) {
  ur_device_usm_access_capability_flags_t device_usm_flags = 0;
  ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, device_usm_flags));
  if (!(device_usm_flags & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
    GTEST_SKIP() << "Device USM is not supported.";
  }

  ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                  allocation_size, &allocation));
  ASSERT_NE(allocation, nullptr);

  ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, allocation));
  ASSERT_SUCCESS(urKernelSetArgValue(kernel, 1, sizeof(data), nullptr, &data));
  Launch1DRange(array_size);

  // Copy the device allocation to a host one so we can validate the results.
  void *host_allocation = nullptr;
  ASSERT_SUCCESS(urUSMHostAlloc(context, nullptr, nullptr, allocation_size,
                                &host_allocation));
  ASSERT_NE(host_allocation, nullptr);
  ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, true, host_allocation, allocation,
                                    allocation_size, 0, nullptr, nullptr));
  ValidateAllocation(host_allocation);
}

TEST_P(urKernelSetArgPointerTest, SuccessShared) {
  ur_device_usm_access_capability_flags_t shared_usm_flags = 0;
  ASSERT_SUCCESS(
      uur::GetDeviceUSMSingleSharedSupport(device, shared_usm_flags));
  if (!(shared_usm_flags & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
    GTEST_SKIP() << "Shared USM is not supported.";
  }

  ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                  allocation_size, &allocation));
  ASSERT_NE(allocation, nullptr);

  ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, allocation));
  ASSERT_SUCCESS(urKernelSetArgValue(kernel, 1, sizeof(data), nullptr, &data));
  Launch1DRange(array_size);
  ValidateAllocation(allocation);
}

struct urKernelSetArgPointerNegativeTest : urKernelSetArgPointerTest {
  // Get any valid allocation we can to test validation of the other parameters.
  void SetUpAllocation() {
    ur_device_usm_access_capability_flags_t host_supported = 0;
    ASSERT_SUCCESS(uur::GetDeviceUSMHostSupport(device, host_supported));
    if (host_supported) {
      ASSERT_SUCCESS(urUSMHostAlloc(context, nullptr, nullptr, allocation_size,
                                    &allocation));
      return;
    }

    ur_device_usm_access_capability_flags_t device_supported = 0;
    ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, device_supported));
    if (device_supported) {
      ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                      allocation_size, &allocation));
      return;
    }

    ur_device_usm_access_capability_flags_t shared_supported = 0;
    ASSERT_SUCCESS(
        uur::GetDeviceUSMSingleSharedSupport(device, shared_supported));
    if (shared_supported) {
      ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                      allocation_size, &allocation));
      return;
    }

    if (!(host_supported || device_supported || shared_supported)) {
      GTEST_SKIP() << "USM is not supported.";
    }
  }

  void SetUp() {
    UUR_RETURN_ON_FATAL_FAILURE(urKernelSetArgPointerTest::SetUp());
    UUR_RETURN_ON_FATAL_FAILURE(SetUpAllocation());
    ASSERT_NE(allocation, nullptr);
  }
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelSetArgPointerNegativeTest);

TEST_P(urKernelSetArgPointerNegativeTest, InvalidNullHandleKernel) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urKernelSetArgPointer(nullptr, 0, nullptr, allocation));
}

TEST_P(urKernelSetArgPointerNegativeTest, InvalidKernelArgumentIndex) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{});

  uint32_t num_kernel_args = 0;
  ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS,
                                 sizeof(num_kernel_args), &num_kernel_args,
                                 nullptr));

  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX,
      urKernelSetArgPointer(kernel, num_kernel_args + 1, nullptr, allocation));
}
