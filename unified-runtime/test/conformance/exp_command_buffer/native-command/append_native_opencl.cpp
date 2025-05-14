// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.h"
#include <CL/cl_ext.h>

// Test using using OpenCL cl_khr_command_buffer to add commands to a native
// OpenCL command-buffer.
struct urOpenCLCommandBufferNativeAppendTest
    : uur::command_buffer::urCommandBufferNativeAppendTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::urCommandBufferNativeAppendTest::SetUp());
    if (backend != UR_BACKEND_OPENCL) {
      GTEST_SKIP() << "Native append test is only supported on OpenCL.";
    }

    // OpenCL doesn't support executing command-buffers inside of other
    // command-buffers
    ur_bool_t subgraph_support{};
    EXPECT_SUCCESS(urDeviceGetInfo(
        device, UR_DEVICE_INFO_COMMAND_BUFFER_SUBGRAPH_SUPPORT_EXP,
        sizeof(ur_bool_t), &subgraph_support, nullptr));
    EXPECT_FALSE(subgraph_support);

    for (auto &buffer : buffers) {
      ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE,
                                       allocation_size, nullptr, &buffer));
      ASSERT_NE(buffer, nullptr);
    }

    // The clCommandCopyBufferKHR function pointer needs queried from the
    // platform as it's an extension entry-point
    cl_platform_id native_platform{};
    ASSERT_SUCCESS(urPlatformGetNativeHandle(
        platform, (ur_native_handle_t *)&native_platform));
    ASSERT_NE(cl_platform_id{}, native_platform);

    clCommandCopyBufferKHR = reinterpret_cast<clCommandCopyBufferKHR_fn>(
        clGetExtensionFunctionAddressForPlatform(native_platform,
                                                 "clCommandCopyBufferKHR"));
    assert(clCommandCopyBufferKHR != nullptr);
  }

  void TearDown() override {
    for (auto &buffer : buffers) {
      if (buffer) {
        EXPECT_SUCCESS(urMemRelease(buffer));
      }
    }

    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::urCommandBufferNativeAppendTest::TearDown());
  }

  clCommandCopyBufferKHR_fn clCommandCopyBufferKHR{};
  std::array<ur_mem_handle_t, 3> buffers = {nullptr, nullptr, nullptr};
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urOpenCLCommandBufferNativeAppendTest);

namespace {
struct InteropData {
  ur_exp_command_buffer_handle_t command_buffer;
  ur_device_handle_t device;
  ur_mem_handle_t src_buffer;
  ur_mem_handle_t dst_buffer;
  clCommandCopyBufferKHR_fn clCommandCopyBufferKHR;
};

// Native command-buffer command is a single cl_mem mem copy command
void interop_func(void *data) {
  InteropData *func_data = reinterpret_cast<InteropData *>(data);
  ASSERT_NE(nullptr, func_data);

  cl_command_buffer_khr native_graph{};
  ASSERT_SUCCESS(urCommandBufferGetNativeHandleExp(
      func_data->command_buffer, (ur_native_handle_t *)&native_graph));
  ASSERT_NE(cl_command_buffer_khr{}, native_graph);

  cl_mem native_src_buffer{};
  ASSERT_SUCCESS(
      urMemGetNativeHandle(func_data->src_buffer, func_data->device,
                           (ur_native_handle_t *)&native_src_buffer));
  ASSERT_NE(cl_mem{}, native_src_buffer);

  cl_mem native_dst_buffer{};
  ASSERT_SUCCESS(
      urMemGetNativeHandle(func_data->dst_buffer, func_data->device,
                           (ur_native_handle_t *)&native_dst_buffer));
  ASSERT_NE(cl_mem{}, native_dst_buffer);

  const auto copy_size =
      uur::command_buffer::urCommandBufferNativeAppendTest::allocation_size;
  auto res = func_data->clCommandCopyBufferKHR(
      native_graph, nullptr, nullptr, native_src_buffer, native_dst_buffer, 0,
      0, copy_size, 0, nullptr, nullptr, nullptr);
  ASSERT_EQ(res, CL_SUCCESS);
}
} // end anonymous namespace

// Test command-buffer with a single native command, which when enqueued has an
// eager UR command as a predecessor and eager UR command as a successor.
TEST_P(urOpenCLCommandBufferNativeAppendTest, Success) {
  auto &src_buffer = buffers[0];
  auto &dst_buffer = buffers[1];
  InteropData data{command_buffer, device, src_buffer, dst_buffer,
                   clCommandCopyBufferKHR};
  ASSERT_SUCCESS(urCommandBufferAppendNativeCommandExp(
      command_buffer, &interop_func, &data, nullptr, 0, nullptr, nullptr));

  ASSERT_SUCCESS(urCommandBufferFinalizeExp(command_buffer));

  ASSERT_SUCCESS(urEnqueueMemBufferFill(queue, src_buffer, &val, sizeof(val), 0,
                                        allocation_size, 0, nullptr, nullptr));

  ASSERT_SUCCESS(
      urEnqueueCommandBufferExp(queue, command_buffer, 0, nullptr, nullptr));

  ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, dst_buffer, true, 0,
                                        allocation_size, host_vec.data(), 0,
                                        nullptr, nullptr));

  for (auto &i : host_vec) {
    ASSERT_EQ(i, val);
  }
}

// Test command-buffer native command with other command-buffer commands as
// predecessors and successors
TEST_P(urOpenCLCommandBufferNativeAppendTest, Dependencies) {
  auto &src_buffer = buffers[0];
  auto &dst_buffer = buffers[1];
  ur_exp_command_buffer_sync_point_t sync_point_1;
  ASSERT_SUCCESS(urCommandBufferAppendMemBufferFillExp(
      command_buffer, src_buffer, &val, sizeof(val), 0, allocation_size, 0,
      nullptr, 0, nullptr, &sync_point_1, nullptr, nullptr));

  InteropData data{command_buffer, device, src_buffer, dst_buffer,
                   clCommandCopyBufferKHR};
  ur_exp_command_buffer_sync_point_t sync_point_2;
  ASSERT_SUCCESS(urCommandBufferAppendNativeCommandExp(
      command_buffer, &interop_func, &data, nullptr, 1, &sync_point_1,
      &sync_point_2));

  auto &copy_buffer = buffers[2];
  ASSERT_SUCCESS(urCommandBufferAppendMemBufferCopyExp(
      command_buffer, dst_buffer, copy_buffer, 0, 0, allocation_size, 0,
      nullptr, 0, nullptr, nullptr, nullptr, nullptr));

  ASSERT_SUCCESS(urCommandBufferFinalizeExp(command_buffer));
  ASSERT_SUCCESS(
      urEnqueueCommandBufferExp(queue, command_buffer, 0, nullptr, nullptr));

  ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, copy_buffer, true, 0,
                                        allocation_size, host_vec.data(), 0,
                                        nullptr, nullptr));
  for (auto &i : host_vec) {
    ASSERT_EQ(i, val);
  }
}
