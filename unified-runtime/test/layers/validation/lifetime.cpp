// Copyright (C) 2023-2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %use-mock %validate lifetime-test | FileCheck %s

#include "fixtures.hpp"

// CHECK: [ RUN      ] urTest.testUrAdapterHandleLifetimeExpectFail
// CHECK-NEXT: <VALIDATION>[ERROR]: There are no valid references to handle {{[0-9xa-fA-F]+}}
TEST_F(urTest, testUrAdapterHandleLifetimeExpectFail) {
  size_t size = 0;
  ur_adapter_handle_t adapter = (ur_adapter_handle_t)0xC0FFEE;
  ur_adapter_info_t info_type = UR_ADAPTER_INFO_BACKEND;
  urAdapterGetInfo(adapter, info_type, 0, nullptr, &size);
}

// CHECK: [ RUN      ] valAdapterTest.testUrAdapterHandleLifetimeExpectSuccess
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NOT: There are no valid references to handle
TEST_F(valAdapterTest, testUrAdapterHandleLifetimeExpectSuccess) {
  size_t size = 0;
  ur_adapter_info_t info_type = UR_ADAPTER_INFO_BACKEND;
  urAdapterGetInfo(adapter, info_type, 0, nullptr, &size);
}

// CHECK: [ RUN      ] valAdapterTest.testUrAdapterHandleTypeMismatchExpectFail
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[ERROR]: There are no valid references to handle {{[0-9xa-fA-F]+}}
TEST_F(valAdapterTest, testUrAdapterHandleTypeMismatchExpectFail) {
  size_t size = 0;
  // Use valid adapter handle with incorrect cast.
  ur_device_handle_t device = (ur_device_handle_t)adapter;
  ur_device_info_t info_type = UR_DEVICE_INFO_BACKEND_RUNTIME_VERSION;
  urDeviceGetInfo(device, info_type, 0, nullptr, &size);
}
