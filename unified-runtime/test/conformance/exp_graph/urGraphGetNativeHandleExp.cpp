// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urGraphGetNativeHandleExpInvalidTest = uur::urMultiQueueTypeTest;

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urGraphGetNativeHandleExpInvalidTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

TEST_P(urGraphGetNativeHandleExpInvalidTest, InvalidNullHandleGraph) {
  ur_native_handle_t nativeGraph = 0;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urGraphGetNativeHandleExp(nullptr, &nativeGraph));
}

using urGraphGetNativeHandleExpTest = uur::urGraphExpTest;

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urGraphGetNativeHandleExpTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

TEST_P(urGraphGetNativeHandleExpTest, Success) {
  ur_native_handle_t nativeGraph = 0;
  if (auto error = urGraphGetNativeHandleExp(graph, &nativeGraph)) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_UNSUPPORTED_FEATURE, error);
  }
}

TEST_P(urGraphGetNativeHandleExpTest, InvalidNullPointerNativeGraph) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urGraphGetNativeHandleExp(graph, nullptr));
}
