// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "urIPCPhysMemHandleExpFixtures.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

using urIPCOpenPhysMemHandleExpTest = urIPCPhysMemHandleTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urIPCOpenPhysMemHandleExpTest);

TEST_P(urIPCOpenPhysMemHandleExpTest, Success) {
  ur_physical_mem_handle_t opened_physical_mem = nullptr;
  ASSERT_SUCCESS(urIPCOpenPhysMemHandleExp(
      context, device, ipc_handle_data, ipc_handle_size, &opened_physical_mem));
  ASSERT_NE(opened_physical_mem, nullptr);
  ASSERT_SUCCESS(urIPCClosePhysMemHandleExp(context, opened_physical_mem));
}

TEST_P(urIPCOpenPhysMemHandleExpTest, InvalidNullHandleContext) {
  ur_physical_mem_handle_t opened_physical_mem = nullptr;
  ASSERT_EQ_RESULT(urIPCOpenPhysMemHandleExp(nullptr, device, ipc_handle_data,
                                             ipc_handle_size,
                                             &opened_physical_mem),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urIPCOpenPhysMemHandleExpTest, InvalidNullHandleDevice) {
  ur_physical_mem_handle_t opened_physical_mem = nullptr;
  ASSERT_EQ_RESULT(urIPCOpenPhysMemHandleExp(context, nullptr, ipc_handle_data,
                                             ipc_handle_size,
                                             &opened_physical_mem),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urIPCOpenPhysMemHandleExpTest, InvalidNullPointerIPCHandleData) {
  ur_physical_mem_handle_t opened_physical_mem = nullptr;
  ASSERT_EQ_RESULT(urIPCOpenPhysMemHandleExp(context, device, nullptr,
                                             ipc_handle_size,
                                             &opened_physical_mem),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urIPCOpenPhysMemHandleExpTest, InvalidNullPointerPhysMem) {
  ASSERT_EQ_RESULT(urIPCOpenPhysMemHandleExp(context, device, ipc_handle_data,
                                             ipc_handle_size, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urIPCOpenPhysMemHandleExpTest, InvalidValue) {
  ur_physical_mem_handle_t opened_physical_mem = nullptr;
  // Pass a size that differs from the real handle data size to trigger
  // UR_RESULT_ERROR_INVALID_VALUE.
  ASSERT_EQ_RESULT(urIPCOpenPhysMemHandleExp(context, device, ipc_handle_data,
                                             ipc_handle_size + 1,
                                             &opened_physical_mem),
                   UR_RESULT_ERROR_INVALID_VALUE);
}

TEST_P(urIPCOpenPhysMemHandleExpTest, InvalidValueStaleHandle) {
  // Obtain a second IPC handle for the same physical memory object and
  // release it right away with urIPCPutPhysMemHandleExp, which closes its
  // underlying file descriptor. urIPCPutPhysMemHandleExp also frees the
  // handle data itself, so save a byte-for-byte copy beforehand and use that
  // copy afterwards. Opening the stale copy must fail because the fd it
  // refers to has already been closed.
  void *stale_handle_data = nullptr;
  size_t stale_handle_size = 0;
  ASSERT_SUCCESS(urIPCGetPhysMemHandleExp(
      context, physical_mem, &stale_handle_data, &stale_handle_size));
  ASSERT_NE(stale_handle_data, nullptr);

  std::vector<uint8_t> handle_copy(stale_handle_size);
  std::memcpy(handle_copy.data(), stale_handle_data, stale_handle_size);

  ASSERT_SUCCESS(urIPCPutPhysMemHandleExp(context, stale_handle_data));

  ur_physical_mem_handle_t opened_physical_mem = nullptr;
  ASSERT_EQ_RESULT(
      urIPCOpenPhysMemHandleExp(context, device, handle_copy.data(),
                                stale_handle_size, &opened_physical_mem),
      UR_RESULT_ERROR_INVALID_VALUE);
}

// Adapter-specific (level_zero_v2) tests that rely on the internal layout of
// the opaque IPC handle data now live in
// unified-runtime/test/adapters/level_zero/v2/urIPCOpenPhysMemHandleExpInvalidHandleData.cpp
