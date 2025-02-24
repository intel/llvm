// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/fixtures.h"
#include "uur/raii.h"

using urContextCreateTest = uur::urDeviceTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urContextCreateTest);

TEST_P(urContextCreateTest, Success) {
  uur::raii::Context context = nullptr;
  ASSERT_SUCCESS(urContextCreate(1, &device, nullptr, context.ptr()));
  ASSERT_NE(nullptr, context);
}

TEST_P(urContextCreateTest, SuccessWithProperties) {
  ur_context_properties_t properties{UR_STRUCTURE_TYPE_CONTEXT_PROPERTIES,
                                     nullptr, 0};
  uur::raii::Context context = nullptr;
  ASSERT_SUCCESS(urContextCreate(1, &device, &properties, context.ptr()));
  ASSERT_NE(nullptr, context);
}

TEST_P(urContextCreateTest, InvalidNullPointerDevices) {
  uur::raii::Context context = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urContextCreate(1, nullptr, nullptr, context.ptr()));
}

TEST_P(urContextCreateTest, InvalidNullPointerContext) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urContextCreate(1, &device, nullptr, nullptr));
}

TEST_P(urContextCreateTest, InvalidEnumeration) {
  ur_context_properties_t properties{UR_STRUCTURE_TYPE_CONTEXT_PROPERTIES,
                                     nullptr, UR_CONTEXT_FLAGS_MASK};
  uur::raii::Context context = nullptr;

  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                   urContextCreate(1, &device, &properties, context.ptr()));
}

using urContextCreateMultiDeviceTest = uur::urAllDevicesTest;
UUR_INSTANTIATE_PLATFORM_TEST_SUITE(urContextCreateMultiDeviceTest);

TEST_P(urContextCreateMultiDeviceTest, Success) {
  if (devices.size() < 2) {
    GTEST_SKIP();
  }
  uur::raii::Context context = nullptr;
  ASSERT_SUCCESS(urContextCreate(static_cast<uint32_t>(devices.size()),
                                 devices.data(), nullptr, context.ptr()));
  ASSERT_NE(nullptr, context);
}
