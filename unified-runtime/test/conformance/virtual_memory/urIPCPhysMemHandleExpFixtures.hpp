// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <uur/fixtures.h>

// Fixture that creates a physical memory object with the IPC-export flag.
struct urIPCPhysMemTest : uur::urVirtualMemGranularityTest {
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

    ur_physical_mem_properties_t properties{
        UR_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES, nullptr,
        UR_PHYSICAL_MEM_FLAG_ENABLE_IPC};
    ur_result_t res =
        urPhysicalMemCreate(context, device, size, &properties, &physical_mem);
    if (res == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
      GTEST_SKIP() << "IPC physical memory is not supported.";
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

// Fixture that also acquires an IPC handle for the physical memory object.
struct urIPCPhysMemHandleTest : urIPCPhysMemTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urIPCPhysMemTest::SetUp());
    ur_result_t res = urIPCGetPhysMemHandleExp(
        context, physical_mem, &ipc_handle_data, &ipc_handle_size);
    if (res == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
      GTEST_SKIP() << "IPC physical memory handle export is not supported.";
    }
    ASSERT_SUCCESS(res);
    ASSERT_NE(ipc_handle_data, nullptr);
    ASSERT_NE(ipc_handle_size, 0U);
  }

  void TearDown() override {
    if (ipc_handle_data) {
      EXPECT_SUCCESS(urIPCPutPhysMemHandleExp(context, ipc_handle_data));
      ipc_handle_data = nullptr;
    }
    urIPCPhysMemTest::TearDown();
  }

  void *ipc_handle_data = nullptr;
  size_t ipc_handle_size = 0;
};

// Fixture that also opens the IPC handle to produce a second physical_mem.
struct urIPCOpenedPhysMemTest : urIPCPhysMemHandleTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urIPCPhysMemHandleTest::SetUp());
    ASSERT_SUCCESS(urIPCOpenPhysMemHandleExp(context, device, ipc_handle_data,
                                             ipc_handle_size,
                                             &opened_physical_mem));
    ASSERT_NE(opened_physical_mem, nullptr);
  }

  void TearDown() override {
    if (opened_physical_mem) {
      EXPECT_SUCCESS(urIPCClosePhysMemHandleExp(context, opened_physical_mem));
      opened_physical_mem = nullptr;
    }
    urIPCPhysMemHandleTest::TearDown();
  }

  ur_physical_mem_handle_t opened_physical_mem = nullptr;
};
