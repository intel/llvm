// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "uur/known_failure.h"
#include <uur/fixtures.h>
#include <uur/known_failure.h>

using urMemReleaseTest = uur::urMemBufferTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urMemReleaseTest);

TEST_P(urMemReleaseTest, Success) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  ASSERT_SUCCESS(urMemRetain(buffer));
  ASSERT_SUCCESS(urMemRelease(buffer));
}

TEST_P(urMemReleaseTest, InvalidNullHandleMem) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE, urMemRelease(nullptr));
}

TEST_P(urMemReleaseTest, CheckReferenceCount) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  uint32_t referenceCount = 0;
  ASSERT_SUCCESS(urMemGetInfo(buffer, UR_MEM_INFO_REFERENCE_COUNT,
                              sizeof(referenceCount), &referenceCount,
                              nullptr));
  ASSERT_EQ(referenceCount, 1);

  ASSERT_SUCCESS(urMemRetain(buffer));
  ASSERT_SUCCESS(urMemGetInfo(buffer, UR_MEM_INFO_REFERENCE_COUNT,
                              sizeof(referenceCount), &referenceCount,
                              nullptr));
  ASSERT_EQ(referenceCount, 2);

  ASSERT_SUCCESS(urMemRelease(buffer));

  ASSERT_SUCCESS(urMemGetInfo(buffer, UR_MEM_INFO_REFERENCE_COUNT,
                              sizeof(referenceCount), &referenceCount,
                              nullptr));
  ASSERT_EQ(referenceCount, 1);
}
