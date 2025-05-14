// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.h"
#include <cuda.h>

// Test using using CUDA-Graph to add commands to a native CUDA command-buffer.
struct urCudaCommandBufferNativeAppendTest
    : uur::command_buffer::urCommandBufferNativeAppendTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::urCommandBufferNativeAppendTest::SetUp());
    if (backend != UR_BACKEND_CUDA) {
      GTEST_SKIP() << "Native append test is only supported on CUDA.";
    }

    // CUDA-Graph supports adds sub-graph nodes to a parent graph
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

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urCudaCommandBufferNativeAppendTest);

namespace {
struct InteropData {
  ur_exp_command_buffer_handle_t command_buffer;
  ur_context_handle_t context;
  void *src;
  void *dst;
};

// Native command-buffer command is a single USM device pointer copy command
void interop_func(void *data) {
  InteropData *func_data = reinterpret_cast<InteropData *>(data);
  ASSERT_NE(nullptr, func_data);

  CUgraph native_graph{};
  ASSERT_SUCCESS(urCommandBufferGetNativeHandleExp(
      func_data->command_buffer, (ur_native_handle_t *)&native_graph));
  ASSERT_NE(CUgraph{}, native_graph);

  CUcontext native_context{};
  ASSERT_SUCCESS(urContextGetNativeHandle(
      func_data->context, (ur_native_handle_t *)&native_context));
  ASSERT_NE(CUcontext{}, native_context);

  CUDA_MEMCPY3D params{};
  params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  params.srcDevice = (CUdeviceptr)func_data->src;
  params.srcHost = nullptr;
  params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  params.dstDevice = (CUdeviceptr)func_data->dst;
  params.dstHost = nullptr;
  params.WidthInBytes =
      uur::command_buffer::urCommandBufferNativeAppendTest::allocation_size;
  params.Height = 1;
  params.Depth = 1;

  CUgraphNode node;
  auto res = cuGraphAddMemcpyNode(&node, native_graph, nullptr, 0, &params,
                                  native_context);
  ASSERT_EQ(res, CUDA_SUCCESS);
}
} // end anonymous namespace

// Test command-buffer with a single native command, which when enqueued has an
// eager UR command as a predecessor and eager UR command as a successor.
TEST_P(urCudaCommandBufferNativeAppendTest, Success) {
  InteropData data{child_cmd_buf, context, src_device_ptr, dst_device_ptr};
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
TEST_P(urCudaCommandBufferNativeAppendTest, Dependencies) {
  ur_exp_command_buffer_sync_point_t sync_point_1;
  ASSERT_SUCCESS(urCommandBufferAppendUSMFillExp(
      command_buffer, src_device_ptr, &val, sizeof(val), allocation_size, 0,
      nullptr, 0, nullptr, &sync_point_1, nullptr, nullptr));

  InteropData data{child_cmd_buf, context, src_device_ptr, dst_device_ptr};
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
