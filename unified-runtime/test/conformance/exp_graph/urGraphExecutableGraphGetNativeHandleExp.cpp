// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urGraphExecutableGraphGetNativeHandleExpInvalidTest =
    uur::urMultiQueueTypeTest;

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urGraphExecutableGraphGetNativeHandleExpInvalidTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

TEST_P(urGraphExecutableGraphGetNativeHandleExpInvalidTest,
       InvalidNullHandleExecutableGraph) {
  ur_native_handle_t nativeExecutableGraph = 0;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urGraphExecutableGraphGetNativeHandleExp(
                       nullptr, &nativeExecutableGraph));
}

using urGraphExecutableGraphGetNativeHandleExpTest =
    uur::urGraphExecutableExpTest;

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urGraphExecutableGraphGetNativeHandleExpTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

TEST_P(urGraphExecutableGraphGetNativeHandleExpTest, Success) {
  ur_native_handle_t nativeExecutableGraph = 0;
  if (auto error = urGraphExecutableGraphGetNativeHandleExp(
          exGraph, &nativeExecutableGraph)) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_UNSUPPORTED_FEATURE, error);
  }
}

TEST_P(urGraphExecutableGraphGetNativeHandleExpTest,
       InvalidNullPointerNativeExecutableGraph) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urGraphExecutableGraphGetNativeHandleExp(exGraph, nullptr));
}
