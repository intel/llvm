// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "urIPCPhysMemHandleExpFixtures.hpp"

using urIPCClosePhysMemHandleExpTest = urIPCOpenedPhysMemTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urIPCClosePhysMemHandleExpTest);

TEST_P(urIPCClosePhysMemHandleExpTest, Success) {
  ASSERT_SUCCESS(urIPCClosePhysMemHandleExp(context, opened_physical_mem));
  // Prevent TearDown from attempting a second close.
  opened_physical_mem = nullptr;
}

TEST_P(urIPCClosePhysMemHandleExpTest, InvalidNullHandleContext) {
  ASSERT_EQ_RESULT(urIPCClosePhysMemHandleExp(nullptr, opened_physical_mem),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urIPCClosePhysMemHandleExpTest, InvalidNullHandlePhysMem) {
  ASSERT_EQ_RESULT(urIPCClosePhysMemHandleExp(context, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}
