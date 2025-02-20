// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/fixtures.h"
#include "uur/known_failure.h"
#include "uur/raii.h"

using urQueueFlushTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urQueueFlushTest);

TEST_P(urQueueFlushTest, Success) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  constexpr size_t buffer_size = 1024;
  uur::raii::Mem buffer = nullptr;
  ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, buffer_size,
                                   nullptr, buffer.ptr()));

  std::vector<uint8_t> data(buffer_size, 42);
  ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, /* blocking */ false, 0,
                                         1024, data.data(), 0, nullptr,
                                         nullptr));

  ASSERT_SUCCESS(urQueueFlush(queue));
}

TEST_P(urQueueFlushTest, InvalidNullHandleQueue) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE, urQueueFlush(nullptr));
}
