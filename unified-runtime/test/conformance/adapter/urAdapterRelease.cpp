// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urAdapterReleaseTest = uur::urAdapterTest;
UUR_INSTANTIATE_ADAPTER_TEST_SUITE(urAdapterReleaseTest);

TEST_P(urAdapterReleaseTest, Success) {
  uint32_t referenceCountBefore = 0;
  ASSERT_SUCCESS(urAdapterRetain(adapter));

  ASSERT_SUCCESS(urAdapterGetInfo(adapter, UR_ADAPTER_INFO_REFERENCE_COUNT,
                                  sizeof(referenceCountBefore),
                                  &referenceCountBefore, nullptr));

  uint32_t referenceCountAfter = 0;
  ASSERT_SUCCESS(urAdapterRelease(adapter));
  ASSERT_SUCCESS(urAdapterGetInfo(adapter, UR_ADAPTER_INFO_REFERENCE_COUNT,
                                  sizeof(referenceCountAfter),
                                  &referenceCountAfter, nullptr));

  ASSERT_LE(referenceCountAfter, referenceCountBefore);
}

TEST_P(urAdapterReleaseTest, InvalidNullHandleAdapter) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urAdapterRelease(nullptr));
}
