// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %with-v2 ./urIPCOpenPhysMemHandleExp-test
// REQUIRES: v2

#include "urIPCPhysMemHandleExpFixtures.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
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

// Mirrors the layout of the opaque handle data produced by the level_zero_v2
// adapter (see unified-runtime/source/adapters/level_zero/v2/physical_mem.hpp
// -- ZeIPCPhysMemHandleData). This IPC feature is currently only implemented
// by that adapter, gated behind UR_DEVICE_INFO_IPC_PHYSICAL_MEMORY_SUPPORT_EXP
// (checked in the fixture's SetUp(), which GTEST_SKIPs otherwise), so
// mirroring its layout here is safe. It lets the tests below exercise
// adapter-internal validation and cross-process-import error paths that
// cannot be reached by treating the handle purely as an opaque blob.
struct TestIPCPhysMemHandleData {
  int Pid;
  int Fd;
  size_t Size;
};

TEST_P(urIPCOpenPhysMemHandleExpTest, InvalidValueCorruptedFields) {
  ASSERT_EQ(ipc_handle_size, sizeof(TestIPCPhysMemHandleData));
  TestIPCPhysMemHandleData corrupted;
  std::memcpy(&corrupted, ipc_handle_data, sizeof(corrupted));
  // A zero size fails the sanity checks performed at the top of
  // urIPCOpenPhysMemHandleExp, before any fd is touched.
  corrupted.Size = 0;

  ur_physical_mem_handle_t opened_physical_mem = nullptr;
  ASSERT_EQ_RESULT(urIPCOpenPhysMemHandleExp(context, device, &corrupted,
                                             sizeof(corrupted),
                                             &opened_physical_mem),
                   UR_RESULT_ERROR_INVALID_VALUE);
}

TEST_P(urIPCOpenPhysMemHandleExpTest, InvalidValueUnknownPid) {
  ASSERT_EQ(ipc_handle_size, sizeof(TestIPCPhysMemHandleData));
  TestIPCPhysMemHandleData corrupted;
  std::memcpy(&corrupted, ipc_handle_data, sizeof(corrupted));
  // Use a PID that is (almost certainly) not in use so the cross-process
  // import path (pidfd_open/pidfd_getfd) is exercised and fails.
  corrupted.Pid = std::numeric_limits<int>::max();

  ur_physical_mem_handle_t opened_physical_mem = nullptr;
  ur_result_t result = urIPCOpenPhysMemHandleExp(
      context, device, &corrupted, sizeof(corrupted), &opened_physical_mem);
  EXPECT_TRUE(result == UR_RESULT_ERROR_INVALID_ARGUMENT ||
              result == UR_RESULT_ERROR_UNSUPPORTED_FEATURE ||
              result == UR_RESULT_ERROR_INVALID_VALUE)
      << "Unexpected result: " << result;
}
