// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.h"
#include <level_zero/ze_api.h>

// Test using using Level-Zero command-lists to add commands to a native
// Level-Zero command-buffer.
struct urL0CommandBufferNativeAppendTest
    : uur::command_buffer::urCommandBufferNativeAppendTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::urCommandBufferNativeAppendTest::SetUp());
    if (backend != UR_BACKEND_LEVEL_ZERO) {
      GTEST_SKIP() << "Native append test is only supported on L0.";
    }

    // Initialize Level Zero driver is required if this test is linked
    // statically with Level Zero loader, the driver will not be init otherwise.
    zeInit(ZE_INIT_FLAG_GPU_ONLY);

    // L0 doesn't support executing command-lists inside of other
    // command-lists
    ur_bool_t subgraph_support{};
    EXPECT_SUCCESS(urDeviceGetInfo(
        device, UR_DEVICE_INFO_COMMAND_BUFFER_SUBGRAPH_SUPPORT_EXP,
        sizeof(ur_bool_t), &subgraph_support, nullptr));
    EXPECT_FALSE(subgraph_support);
  }
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urL0CommandBufferNativeAppendTest);

namespace {
struct InteropData {
  ur_exp_command_buffer_handle_t command_buffer;
  void *src;
  void *dst;
};

// Native command-buffer command is a single USM device pointer copy command
void interop_func(void *data) {
  InteropData *func_data = reinterpret_cast<InteropData *>(data);
  ASSERT_NE(nullptr, func_data);

  ze_command_list_handle_t native_graph{};
  ASSERT_SUCCESS(urCommandBufferGetNativeHandleExp(
      func_data->command_buffer, (ur_native_handle_t *)&native_graph));
  ASSERT_NE(ze_command_list_handle_t{}, native_graph);

  const auto copy_size =
      uur::command_buffer::urCommandBufferNativeAppendTest::allocation_size;
  auto res = zeCommandListAppendMemoryCopy(native_graph, func_data->dst,
                                           func_data->src, copy_size, nullptr,
                                           0, nullptr);
  ASSERT_EQ(res, ZE_RESULT_SUCCESS);
}
} // end anonymous namespace

// Test command-buffer with a single native command, which when enqueued has an
// eager UR command as a predecessor and eager UR command as a successor.
TEST_P(urL0CommandBufferNativeAppendTest, Success) {
  InteropData data{command_buffer, src_device_ptr, dst_device_ptr};
  ASSERT_SUCCESS(urCommandBufferAppendNativeCommandExp(
      command_buffer, &interop_func, &data, nullptr, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urCommandBufferFinalizeExp(command_buffer));

  ASSERT_SUCCESS(urEnqueueUSMFill(queue, src_device_ptr, sizeof(val), &val,
                                  allocation_size, 0, nullptr, nullptr));

  ASSERT_SUCCESS(
      urEnqueueCommandBufferExp(queue, command_buffer, 0, nullptr, nullptr));

  ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, host_vec.data(),
                                    dst_device_ptr, allocation_size, 0, nullptr,
                                    nullptr));

  urQueueFinish(queue);
  for (auto &i : host_vec) {
    ASSERT_EQ(i, val);
  }
}

// Test command-buffer native command with other command-buffer commands as
// predecessors and successors
TEST_P(urL0CommandBufferNativeAppendTest, Dependencies) {
  ur_exp_command_buffer_sync_point_t sync_point_1;
  ASSERT_SUCCESS(urCommandBufferAppendUSMFillExp(
      command_buffer, src_device_ptr, &val, sizeof(val), allocation_size, 0,
      nullptr, 0, nullptr, &sync_point_1, nullptr, nullptr));

  InteropData data{command_buffer, src_device_ptr, dst_device_ptr};
  ur_exp_command_buffer_sync_point_t sync_point_2;
  ASSERT_SUCCESS(urCommandBufferAppendNativeCommandExp(
      command_buffer, &interop_func, &data, nullptr, 1, &sync_point_1,
      &sync_point_2));

  ASSERT_SUCCESS(urCommandBufferAppendUSMMemcpyExp(
      command_buffer, host_vec.data(), dst_device_ptr, allocation_size, 1,
      &sync_point_2, 0, nullptr, nullptr, nullptr, nullptr));

  ASSERT_SUCCESS(urCommandBufferFinalizeExp(command_buffer));
  ASSERT_SUCCESS(
      urEnqueueCommandBufferExp(queue, command_buffer, 0, nullptr, nullptr));

  urQueueFinish(queue);
  for (auto &i : host_vec) {
    ASSERT_EQ(i, val);
  }
}
