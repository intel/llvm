// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "urIPCPhysMemHandleExpFixtures.hpp"

using urIPCGetPhysMemHandleExpTest = urIPCPhysMemTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urIPCGetPhysMemHandleExpTest);

TEST_P(urIPCGetPhysMemHandleExpTest, Success) {
  void *ipc_handle_data = nullptr;
  size_t ipc_handle_size = 0;
  ur_result_t res = urIPCGetPhysMemHandleExp(
      context, physical_mem, &ipc_handle_data, &ipc_handle_size);
  if (res == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    GTEST_SKIP() << "IPC physical memory handle export is not supported.";
  }
  ASSERT_SUCCESS(res);
  ASSERT_NE(ipc_handle_data, nullptr);
  ASSERT_NE(ipc_handle_size, 0U);
  ASSERT_SUCCESS(urIPCPutPhysMemHandleExp(context, ipc_handle_data));
}

TEST_P(urIPCGetPhysMemHandleExpTest, InvalidNullHandleContext) {
  void *ipc_handle_data = nullptr;
  size_t ipc_handle_size = 0;
  ASSERT_EQ_RESULT(urIPCGetPhysMemHandleExp(nullptr, physical_mem,
                                            &ipc_handle_data, &ipc_handle_size),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urIPCGetPhysMemHandleExpTest, InvalidNullHandlePhysMem) {
  void *ipc_handle_data = nullptr;
  size_t ipc_handle_size = 0;
  ASSERT_EQ_RESULT(urIPCGetPhysMemHandleExp(context, nullptr, &ipc_handle_data,
                                            &ipc_handle_size),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urIPCGetPhysMemHandleExpTest, InvalidNullPointerIPCHandleData) {
  size_t ipc_handle_size = 0;
  ASSERT_EQ_RESULT(urIPCGetPhysMemHandleExp(context, physical_mem, nullptr,
                                            &ipc_handle_size),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urIPCGetPhysMemHandleExpTest, InvalidNullPointerIPCHandleDataSize) {
  void *ipc_handle_data = nullptr;
  ASSERT_EQ_RESULT(urIPCGetPhysMemHandleExp(context, physical_mem,
                                            &ipc_handle_data, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

// Fixture that creates a physical memory object WITHOUT the IPC-export flag.
struct urIPCGetPhysMemHandleExpNoIpcTest : uur::urVirtualMemGranularityTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urVirtualMemGranularityTest::SetUp());
    size = granularity * 256;

    ur_bool_t ipc_support = false;
    ASSERT_SUCCESS(
        urDeviceGetInfo(device, UR_DEVICE_INFO_IPC_PHYSICAL_MEMORY_SUPPORT_EXP,
                        sizeof(ur_bool_t), &ipc_support, nullptr));
    if (!ipc_support) {
      GTEST_SKIP() << "IPC physical memory is not supported.";
    }

    // Create physical memory without ENABLE_IPC flag.
    ur_result_t res =
        urPhysicalMemCreate(context, device, size, nullptr, &physical_mem);
    if (res == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
      GTEST_SKIP() << "Physical memory creation is not supported.";
    }
    ASSERT_SUCCESS(res);
    ASSERT_NE(physical_mem, nullptr);
  }

  void TearDown() override {
    if (physical_mem) {
      EXPECT_SUCCESS(urPhysicalMemRelease(physical_mem));
    }
    uur::urVirtualMemGranularityTest::TearDown();
  }

  size_t size = 0;
  ur_physical_mem_handle_t physical_mem = nullptr;
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urIPCGetPhysMemHandleExpNoIpcTest);

TEST_P(urIPCGetPhysMemHandleExpNoIpcTest, InvalidArgumentNoIpcFlag) {
  // Exporting a handle for memory not created with ENABLE_IPC must fail.
  void *ipc_handle_data = nullptr;
  size_t ipc_handle_size = 0;
  ASSERT_EQ_RESULT(urIPCGetPhysMemHandleExp(context, physical_mem,
                                            &ipc_handle_data, &ipc_handle_size),
                   UR_RESULT_ERROR_INVALID_ARGUMENT);
}
