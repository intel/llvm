// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urHipQueueGetNativeHandleTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urHipQueueGetNativeHandleTest);

TEST_P(urHipQueueGetNativeHandleTest, UseDefaultStreamAlone) {
  hipStream_t Stream;
  ur_queue_properties_t props = {
      /*.stype =*/UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
      /*.pNext =*/nullptr,
      /*.flags =*/UR_QUEUE_FLAG_USE_DEFAULT_STREAM,
  };
  ASSERT_SUCCESS(urQueueCreate(context, device, &props, &queue));
  ASSERT_SUCCESS(
      urQueueGetNativeHandle(queue, nullptr, (ur_native_handle_t *)&Stream));
  unsigned int Flags = 0;
  ASSERT_SUCCESS_HIP(hipStreamGetFlags(Stream, &Flags));
  ASSERT_EQ(Flags, static_cast<unsigned int>(hipStreamDefault));
}

TEST_P(urHipQueueGetNativeHandleTest, SyncWithDefaultStreamAlone) {
  hipStream_t Stream;
  ur_queue_properties_t props = {
      /*.stype =*/UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
      /*.pNext =*/nullptr,
      /*.flags =*/UR_QUEUE_FLAG_SYNC_WITH_DEFAULT_STREAM,
  };
  ASSERT_SUCCESS(urQueueCreate(context, device, &props, &queue));
  ASSERT_SUCCESS(
      urQueueGetNativeHandle(queue, nullptr, (ur_native_handle_t *)&Stream));
  unsigned int Flags = 0;
  ASSERT_SUCCESS_HIP(hipStreamGetFlags(Stream, &Flags));
  ASSERT_EQ(Flags, static_cast<unsigned int>(hipStreamNonBlocking));
}

TEST_P(urHipQueueGetNativeHandleTest, UseDefaultStreamCombinedWithPriority) {
  hipStream_t Stream;
  ur_queue_properties_t props = {
      /*.stype =*/UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
      /*.pNext =*/nullptr,
      /*.flags =*/UR_QUEUE_FLAG_USE_DEFAULT_STREAM |
          UR_QUEUE_FLAG_PRIORITY_HIGH,
  };
  ASSERT_SUCCESS(urQueueCreate(context, device, &props, &queue));
  ASSERT_SUCCESS(
      urQueueGetNativeHandle(queue, nullptr, (ur_native_handle_t *)&Stream));
  unsigned int Flags = 0;
  ASSERT_SUCCESS_HIP(hipStreamGetFlags(Stream, &Flags));
  ASSERT_EQ(Flags, static_cast<unsigned int>(hipStreamDefault));
}

TEST_P(urHipQueueGetNativeHandleTest,
       SyncWithDefaultStreamCombinedWithPriority) {
  hipStream_t Stream;
  ur_queue_properties_t props = {
      /*.stype =*/UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
      /*.pNext =*/nullptr,
      /*.flags =*/UR_QUEUE_FLAG_SYNC_WITH_DEFAULT_STREAM |
          UR_QUEUE_FLAG_PRIORITY_LOW,
  };
  ASSERT_SUCCESS(urQueueCreate(context, device, &props, &queue));
  ASSERT_SUCCESS(
      urQueueGetNativeHandle(queue, nullptr, (ur_native_handle_t *)&Stream));
  unsigned int Flags = 0;
  ASSERT_SUCCESS_HIP(hipStreamGetFlags(Stream, &Flags));
  ASSERT_EQ(Flags, static_cast<unsigned int>(hipStreamNonBlocking));
}
