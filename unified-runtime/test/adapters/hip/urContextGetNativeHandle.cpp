// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urHipContextGetNativeHandleTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urHipContextGetNativeHandleTest);

TEST_P(urHipContextGetNativeHandleTest, Success) {
  ur_native_handle_t native_context = 0;
  auto status = urContextGetNativeHandle(context, &native_context);
  ASSERT_EQ(status, UR_RESULT_ERROR_UNSUPPORTED_FEATURE);
}
