// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/checks.h"
#include <uur/fixtures.h>

using urKernelRetainTest = uur::urKernelTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelRetainTest);

TEST_P(urKernelRetainTest, Success) {
  ASSERT_SUCCESS(urKernelRetain(kernel));
  EXPECT_SUCCESS(urKernelRelease(kernel));
}

TEST_P(urKernelRetainTest, InvalidNullHandleKernel) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urKernelRetain(nullptr));
}

TEST_P(urKernelRetainTest, CheckReferenceCount) {
  uint32_t referenceCount = 0;
  ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_REFERENCE_COUNT,
                                 sizeof(referenceCount), &referenceCount,
                                 nullptr));
  ASSERT_EQ(referenceCount, 1);

  ASSERT_SUCCESS(urKernelRetain(kernel));

  ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_REFERENCE_COUNT,
                                 sizeof(referenceCount), &referenceCount,
                                 nullptr));
  ASSERT_EQ(referenceCount, 2);

  ASSERT_SUCCESS(urKernelRelease(kernel));

  ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_REFERENCE_COUNT,
                                 sizeof(referenceCount), &referenceCount,
                                 nullptr));
  ASSERT_EQ(referenceCount, 1);
}
