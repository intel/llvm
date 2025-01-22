// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "../../enqueue/helpers.h"
#include <uur/fixtures.h>
#include <uur/known_failure.h>
#include <uur/raii.h>

using urEnqueueMemBufferMapTestWithParamL0 =
    uur::urMemBufferQueueTestWithParam<uur::mem_buffer_test_parameters_t>;

UUR_DEVICE_TEST_SUITE_P(
    urEnqueueMemBufferMapTestWithParamL0,
    ::testing::ValuesIn(uur::mem_buffer_test_parameters),
    uur::printMemBufferTestString<urEnqueueMemBufferMapTestWithParamL0>);

TEST_P(urEnqueueMemBufferMapTestWithParamL0, MapWithHostPtr) {
  uur::raii::Mem buffer;

  char *ptr = new char[4096];

  ur_buffer_properties_t props;
  props.pHost = ptr;

  ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_USE_HOST_POINTER, 4096,
                                   &props, buffer.ptr()));

  void *mappedPtr = nullptr;
  ASSERT_SUCCESS(urEnqueueMemBufferMap(
      queue, buffer.get(), true, UR_MAP_FLAG_READ | UR_MAP_FLAG_WRITE, 0, 4096,
      0, nullptr, nullptr, reinterpret_cast<void **>(&mappedPtr)));

  ASSERT_EQ(ptr, mappedPtr);

  delete[] ptr;
}
