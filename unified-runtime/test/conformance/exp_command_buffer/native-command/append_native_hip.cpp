// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.h"

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

// Test using using HIP-Graph to add commands to a native HIP command-buffer.
struct urHipCommandBufferNativeAppendTest
    : uur::command_buffer::urCommandBufferNativeAppendTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::urCommandBufferNativeAppendTest::SetUp());
    if (backend != UR_BACKEND_HIP) {
      GTEST_SKIP() << "Native append test is only supported on HIP.";
    }

    // HIP-Graph supports adds sub-graph nodes to a parent graph
    ur_bool_t subgraph_support = false;
    EXPECT_SUCCESS(urDeviceGetInfo(
        device, UR_DEVICE_INFO_COMMAND_BUFFER_SUBGRAPH_SUPPORT_EXP,
        sizeof(ur_bool_t), &subgraph_support, nullptr));
    EXPECT_TRUE(subgraph_support);

    // Create a non-updatable graph as a child graph
    ur_exp_command_buffer_desc_t desc{
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC /*stype*/, nullptr /*pnext*/,
        false /* updatable */, false /* in-order */, false /* profilable*/
    };
    UUR_RETURN_ON_FATAL_FAILURE(
        urCommandBufferCreateExp(context, device, &desc, &child_cmd_buf));
  }

  void TearDown() override {
    if (child_cmd_buf) {
      EXPECT_SUCCESS(urCommandBufferReleaseExp(child_cmd_buf));
    }
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::urCommandBufferNativeAppendTest::TearDown());
  }

  ur_exp_command_buffer_handle_t child_cmd_buf = nullptr;
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urHipCommandBufferNativeAppendTest);

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

  hipGraph_t native_graph{};
  ASSERT_SUCCESS(urCommandBufferGetNativeHandleExp(
      func_data->command_buffer, (ur_native_handle_t *)&native_graph));
  ASSERT_NE(hipGraph_t{}, native_graph);

  const auto copy_size =
      uur::command_buffer::urCommandBufferNativeAppendTest::allocation_size;
  hipGraphNode_t node;
  auto res =
      hipGraphAddMemcpyNode1D(&node, native_graph, nullptr, 0, func_data->dst,
                              func_data->src, copy_size, hipMemcpyDefault);
  ASSERT_EQ(res, hipSuccess);
}
} // end anonymous namespace

// Test command-buffer with a single native command, which when enqueued has an
// eager UR command as a predecessor and eager UR command as a successor.
TEST_P(urHipCommandBufferNativeAppendTest, Success) {
  InteropData data{child_cmd_buf, src_device_ptr, dst_device_ptr};
  ASSERT_SUCCESS(urCommandBufferAppendNativeCommandExp(
      command_buffer, &interop_func, &data, child_cmd_buf, 0, nullptr,
      nullptr));

  ASSERT_SUCCESS(urCommandBufferFinalizeExp(command_buffer));

  ASSERT_SUCCESS(urEnqueueUSMFill(queue, src_device_ptr, sizeof(val), &val,
                                  allocation_size, 0, nullptr, nullptr));
  ASSERT_SUCCESS(
      urEnqueueCommandBufferExp(queue, command_buffer, 0, nullptr, nullptr));

  ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, true, host_vec.data(),
                                    dst_device_ptr, allocation_size, 0, nullptr,
                                    nullptr));
  for (auto &i : host_vec) {
    ASSERT_EQ(i, val);
  }
}

// Test command-buffer native command with other command-buffer commands as
// predecessors and successors
TEST_P(urHipCommandBufferNativeAppendTest, Dependencies) {
  ur_exp_command_buffer_sync_point_t sync_point_1;
  ASSERT_SUCCESS(urCommandBufferAppendUSMFillExp(
      command_buffer, src_device_ptr, &val, sizeof(val), allocation_size, 0,
      nullptr, 0, nullptr, &sync_point_1, nullptr, nullptr));

  InteropData data{child_cmd_buf, src_device_ptr, dst_device_ptr};
  ur_exp_command_buffer_sync_point_t sync_point_2;
  ASSERT_SUCCESS(urCommandBufferAppendNativeCommandExp(
      command_buffer, &interop_func, &data, child_cmd_buf, 1, &sync_point_1,
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
