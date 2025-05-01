// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ur_api.h"
#include "uur/checks.h"
#include "uur/raii.h"
#include "ze_api.h"
#include <uur/fixtures.h>

using urMemBufferCreateWithNativeHandleTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urMemBufferCreateWithNativeHandleTest);

TEST_P(urMemBufferCreateWithNativeHandleTest, SharedBufferIsUsedDirectly) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

  // Initialize Level Zero driver is required if this test is linked statically
  // with Level Zero loader, the driver will not be init otherwise.
  zeInit(ZE_INIT_FLAG_GPU_ONLY);

  ur_native_handle_t nativeContext;
  ASSERT_SUCCESS(urContextGetNativeHandle(context, &nativeContext));

  ur_native_handle_t nativeDevice;
  ASSERT_SUCCESS(urDeviceGetNativeHandle(device, &nativeDevice));

  ze_device_mem_alloc_desc_t DeviceDesc = {};
  DeviceDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  DeviceDesc.ordinal = 0;
  DeviceDesc.flags = 0;
  DeviceDesc.pNext = nullptr;

  ze_host_mem_alloc_desc_t HostDesc = {};
  HostDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
  HostDesc.pNext = nullptr;
  HostDesc.flags = 0;

  void *SharedBuffer = nullptr;
  ASSERT_EQ(
      zeMemAllocShared(reinterpret_cast<ze_context_handle_t>(nativeContext),
                       &DeviceDesc, &HostDesc, 12 * sizeof(int), 1, nullptr,
                       &SharedBuffer),
      ZE_RESULT_SUCCESS);

  uur::raii::Mem buffer;
  ASSERT_SUCCESS(urMemBufferCreateWithNativeHandle(
      reinterpret_cast<ur_native_handle_t>(SharedBuffer), context, nullptr,
      buffer.ptr()));

  void *mappedPtr;
  ASSERT_SUCCESS(urEnqueueMemBufferMap(queue, buffer.get(), true, 0, 0,
                                       12 * sizeof(int), 0, nullptr, nullptr,
                                       &mappedPtr));

  ASSERT_EQ(mappedPtr, SharedBuffer);
  ASSERT_EQ(zeMemFree(reinterpret_cast<ze_context_handle_t>(nativeContext),
                      SharedBuffer),
            ZE_RESULT_SUCCESS);
}
