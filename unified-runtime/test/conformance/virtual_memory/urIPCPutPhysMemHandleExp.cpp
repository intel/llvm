// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "urIPCPhysMemHandleExpFixtures.hpp"

using urIPCPutPhysMemHandleExpTest = urIPCPhysMemHandleTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urIPCPutPhysMemHandleExpTest);

TEST_P(urIPCPutPhysMemHandleExpTest, Success) {
  ASSERT_SUCCESS(urIPCPutPhysMemHandleExp(context, ipc_handle_data));
  // Prevent TearDown from attempting a second put.
  ipc_handle_data = nullptr;
}

TEST_P(urIPCPutPhysMemHandleExpTest, InvalidNullHandleContext) {
  ASSERT_EQ_RESULT(urIPCPutPhysMemHandleExp(nullptr, ipc_handle_data),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urIPCPutPhysMemHandleExpTest, InvalidNullPointerIPCHandleData) {
  ASSERT_EQ_RESULT(urIPCPutPhysMemHandleExp(context, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}
