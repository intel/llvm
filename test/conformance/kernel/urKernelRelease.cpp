// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urKernelReleaseTest = uur::urKernelTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelReleaseTest);

TEST_P(urKernelReleaseTest, Success) {
  ASSERT_SUCCESS(urKernelRetain(kernel));
  ASSERT_SUCCESS(urKernelRelease(kernel));
}

TEST_P(urKernelReleaseTest, KernelReleaseAfterProgramRelease) {
  ASSERT_SUCCESS(urKernelRetain(kernel));
  ASSERT_SUCCESS(urProgramRelease(program));
  program = nullptr;
  ASSERT_SUCCESS(urKernelRelease(kernel));
}

TEST_P(urKernelReleaseTest, InvalidNullHandleKernel) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urKernelRelease(nullptr));
}

TEST_P(urKernelReleaseTest, CheckReferenceCount) {
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
