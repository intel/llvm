// Copyright (C) 2022-2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include "queue.hpp"

using urCudaQueueGetNativeHandleTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urCudaQueueGetNativeHandleTest);

TEST_P(urCudaQueueGetNativeHandleTest, Success) {
  CUstream Stream;
  ASSERT_SUCCESS(
      urQueueGetNativeHandle(queue, nullptr, (ur_native_handle_t *)&Stream));
  ASSERT_SUCCESS_CUDA(cuStreamSynchronize(Stream));
}

TEST_P(urCudaQueueGetNativeHandleTest, OutOfOrder) {
  CUstream Stream;
  ur_queue_properties_t props = {
      /*.stype =*/UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
      /*.pNext =*/nullptr,
      /*.flags =*/UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE,
  };
  ASSERT_SUCCESS(urQueueCreate(context, device, &props, &queue));
  ASSERT_SUCCESS(
      urQueueGetNativeHandle(queue, nullptr, (ur_native_handle_t *)&Stream));
  ASSERT_SUCCESS_CUDA(cuStreamSynchronize(Stream));
}

TEST_P(urCudaQueueGetNativeHandleTest, ScopedStream) {
  CUstream Stream1, Stream2;
  ur_queue_properties_t props = {
      /*.stype =*/UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
      /*.pNext =*/nullptr,
      /*.flags =*/UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE,
  };
  ur_queue_handle_t OutOfOrderQueue;
  ASSERT_SUCCESS(urQueueCreate(context, device, &props, &OutOfOrderQueue));
  ASSERT_SUCCESS(urQueueGetNativeHandle(OutOfOrderQueue, nullptr,
                                        (ur_native_handle_t *)&Stream1));
  ASSERT_SUCCESS(urQueueGetNativeHandle(OutOfOrderQueue, nullptr,
                                        (ur_native_handle_t *)&Stream2));

  // We might want to remove this assertion at some point. This is just
  // testing current implementated behaviour that getting the native
  // OutOfOrderQueue will call `getNextComputeStream`
  ASSERT_NE(Stream1, Stream2);

  {
    ScopedStream ActiveStream(OutOfOrderQueue, 0, nullptr);

    ASSERT_SUCCESS(urQueueGetNativeHandle(OutOfOrderQueue, nullptr,
                                          (ur_native_handle_t *)&Stream1));
    ASSERT_SUCCESS(urQueueGetNativeHandle(OutOfOrderQueue, nullptr,
                                          (ur_native_handle_t *)&Stream2));
    ASSERT_EQ(Stream1, Stream2);
  }

  // Go back to returning new streams each time
  ASSERT_SUCCESS(urQueueGetNativeHandle(OutOfOrderQueue, nullptr,
                                        (ur_native_handle_t *)&Stream1));
  ASSERT_SUCCESS(urQueueGetNativeHandle(OutOfOrderQueue, nullptr,
                                        (ur_native_handle_t *)&Stream2));
  ASSERT_NE(Stream1, Stream2);
}
