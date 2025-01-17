// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include <umf/memory_pool.h>
#include <umf/memory_provider.h>
#include <uur/fixtures.h>

#ifndef ASSERT_UMF_SUCCESS
#define ASSERT_UMF_SUCCESS(ACTUAL) ASSERT_EQ(ACTUAL, UMF_RESULT_SUCCESS)
#endif

using urL0IpcTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urL0IpcTest);

TEST_P(urL0IpcTest, SuccessHostL0Ipc) {
  ur_device_usm_access_capability_flags_t hostUSMSupport = 0;
  ASSERT_SUCCESS(uur::GetDeviceUSMHostSupport(device, hostUSMSupport));
  if (!hostUSMSupport) {
    GTEST_SKIP() << "Host USM is not supported.";
  }

  void *ptr = nullptr;
  size_t allocSize = sizeof(int);
  ASSERT_SUCCESS(urUSMHostAlloc(context, nullptr, nullptr, allocSize, &ptr));
  ASSERT_NE(ptr, nullptr);

  umf_memory_pool_handle_t umfPool = umfPoolByPtr(ptr);
  ASSERT_NE(umfPool, nullptr);

  umf_memory_provider_handle_t umfProvider = nullptr;
  ASSERT_UMF_SUCCESS(umfPoolGetMemoryProvider(umfPool, &umfProvider));

  size_t ipcHandleSize = 0;
  ASSERT_UMF_SUCCESS(
      umfMemoryProviderGetIPCHandleSize(umfProvider, &ipcHandleSize));

  void *ipcHandle = nullptr;
  ASSERT_UMF_SUCCESS(
      umfMemoryProviderAlloc(umfProvider, ipcHandleSize, 0, &ipcHandle));
  ASSERT_UMF_SUCCESS(
      umfMemoryProviderGetIPCHandle(umfProvider, ptr, allocSize, ipcHandle));

  ASSERT_UMF_SUCCESS(umfMemoryProviderPutIPCHandle(umfProvider, ipcHandle));

  ASSERT_UMF_SUCCESS(
      umfMemoryProviderFree(umfProvider, ipcHandle, ipcHandleSize));

  ASSERT_SUCCESS(urUSMFree(context, ptr));
}
