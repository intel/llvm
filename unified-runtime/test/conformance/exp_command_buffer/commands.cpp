// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include <array>

struct urCommandBufferCommandsTest
    : uur::command_buffer::urCommandBufferExpTest {

  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::urCommandBufferExpTest::SetUp());

    // Allocate USM pointers
    for (auto &device_ptr : device_ptrs) {
      ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                      allocation_size, &device_ptr));
      ASSERT_NE(device_ptr, nullptr);
    }

    for (auto &buffer : buffers) {
      ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE,
                                       allocation_size, nullptr, &buffer));

      ASSERT_NE(buffer, nullptr);
    }
  }

  void TearDown() override {
    for (auto &device_ptr : device_ptrs) {
      if (device_ptr) {
        EXPECT_SUCCESS(urUSMFree(context, device_ptr));
      }
    }

    for (auto &buffer : buffers) {
      if (buffer) {
        EXPECT_SUCCESS(urMemRelease(buffer));
      }
    }

    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::urCommandBufferExpTest::TearDown());
  }

  static constexpr unsigned elements = 16;
  static constexpr size_t allocation_size = elements * sizeof(uint32_t);

  std::array<void *, 2> device_ptrs = {nullptr, nullptr};
  std::array<ur_mem_handle_t, 2> buffers = {nullptr, nullptr};
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urCommandBufferCommandsTest);

TEST_P(urCommandBufferCommandsTest, urCommandBufferAppendUSMMemcpyExp) {
  ASSERT_SUCCESS(urCommandBufferAppendUSMMemcpyExp(
      cmd_buf_handle, device_ptrs[0], device_ptrs[1], allocation_size, 0,
      nullptr, 0, nullptr, nullptr, nullptr, nullptr));
}

TEST_P(urCommandBufferCommandsTest, urCommandBufferAppendUSMFillExp) {
  uint32_t pattern = 42;
  ASSERT_SUCCESS(urCommandBufferAppendUSMFillExp(
      cmd_buf_handle, device_ptrs[0], &pattern, sizeof(pattern),
      allocation_size, 0, nullptr, 0, nullptr, nullptr, nullptr, nullptr));
}

TEST_P(urCommandBufferCommandsTest, urCommandBufferAppendMemBufferCopyExp) {
  ASSERT_SUCCESS(urCommandBufferAppendMemBufferCopyExp(
      cmd_buf_handle, buffers[0], buffers[1], 0, 0, allocation_size, 0, nullptr,
      0, nullptr, nullptr, nullptr, nullptr));
}

TEST_P(urCommandBufferCommandsTest, urCommandBufferAppendMemBufferCopyRectExp) {
  ur_rect_offset_t origin{0, 0, 0};
  ur_rect_region_t region{4, 4, 1};
  ASSERT_SUCCESS(urCommandBufferAppendMemBufferCopyRectExp(
      cmd_buf_handle, buffers[0], buffers[1], origin, origin, region, 4, 16, 4,
      16, 0, nullptr, 0, nullptr, nullptr, nullptr, nullptr));
}

TEST_P(urCommandBufferCommandsTest, urCommandBufferAppendMemBufferReadExp) {
  // No buffer read command in cl_khr_command_buffer
  // See https://github.com/KhronosGroup/OpenCL-Docs/issues/1281
  UUR_KNOWN_FAILURE_ON(uur::OpenCL{});

  std::array<uint32_t, elements> host_data{};
  ASSERT_SUCCESS(urCommandBufferAppendMemBufferReadExp(
      cmd_buf_handle, buffers[0], 0, allocation_size, host_data.data(), 0,
      nullptr, 0, nullptr, nullptr, nullptr, nullptr));
}

TEST_P(urCommandBufferCommandsTest, urCommandBufferAppendMemBufferReadRectExp) {
  // No buffer read command in cl_khr_command_buffer
  // See https://github.com/KhronosGroup/OpenCL-Docs/issues/1281
  UUR_KNOWN_FAILURE_ON(uur::OpenCL{});

  std::array<uint32_t, elements> host_data{};
  ur_rect_offset_t origin{0, 0, 0};
  ur_rect_region_t region{4, 4, 1};
  ASSERT_SUCCESS(urCommandBufferAppendMemBufferReadRectExp(
      cmd_buf_handle, buffers[0], origin, origin, region, 4, 16, 4, 16,
      host_data.data(), 0, nullptr, 0, nullptr, nullptr, nullptr, nullptr));
}

TEST_P(urCommandBufferCommandsTest, urCommandBufferAppendMemBufferWriteExp) {
  // No buffer write command in cl_khr_command_buffer
  // See https://github.com/KhronosGroup/OpenCL-Docs/issues/1281
  UUR_KNOWN_FAILURE_ON(uur::OpenCL{});

  std::array<uint32_t, elements> host_data{};
  ASSERT_SUCCESS(urCommandBufferAppendMemBufferWriteExp(
      cmd_buf_handle, buffers[0], 0, allocation_size, host_data.data(), 0,
      nullptr, 0, nullptr, nullptr, nullptr, nullptr));
}

TEST_P(urCommandBufferCommandsTest,
       urCommandBufferAppendMemBufferWriteRectExp) {
  // No buffer write command in cl_khr_command_buffer
  // See https://github.com/KhronosGroup/OpenCL-Docs/issues/1281
  UUR_KNOWN_FAILURE_ON(uur::OpenCL{});

  std::array<uint32_t, elements> host_data{};
  ur_rect_offset_t origin{0, 0, 0};
  ur_rect_region_t region{4, 4, 1};
  ASSERT_SUCCESS(urCommandBufferAppendMemBufferWriteRectExp(
      cmd_buf_handle, buffers[0], origin, origin, region, 4, 16, 4, 16,
      host_data.data(), 0, nullptr, 0, nullptr, nullptr, nullptr, nullptr));
}

TEST_P(urCommandBufferCommandsTest, urCommandBufferAppendMemBufferFillExp) {
  uint32_t pattern = 42;
  ASSERT_SUCCESS(urCommandBufferAppendMemBufferFillExp(
      cmd_buf_handle, buffers[0], &pattern, sizeof(pattern), 0, allocation_size,
      0, nullptr, 0, nullptr, nullptr, nullptr, nullptr));
}

TEST_P(urCommandBufferCommandsTest, urCommandBufferAppendUSMPrefetchExp) {
  // No Prefetch command in cl_khr_command_buffer
  UUR_KNOWN_FAILURE_ON(uur::OpenCL{});

  ASSERT_SUCCESS(urCommandBufferAppendUSMPrefetchExp(
      cmd_buf_handle, device_ptrs[0], allocation_size, 0, 0, nullptr, 0,
      nullptr, nullptr, nullptr, nullptr));
}

TEST_P(urCommandBufferCommandsTest, urCommandBufferAppendUSMAdviseExp) {
  // No advise command in cl_khr_command_buffer
  UUR_KNOWN_FAILURE_ON(uur::OpenCL{});

  ASSERT_SUCCESS(urCommandBufferAppendUSMAdviseExp(
      cmd_buf_handle, device_ptrs[0], allocation_size, 0, 0, nullptr, 0,
      nullptr, nullptr, nullptr, nullptr));
}

struct urCommandBufferAppendKernelLaunchExpTest
    : uur::command_buffer::urCommandBufferExpExecutionTest {
  virtual void SetUp() override {
    program_name = "saxpy_usm";
    UUR_RETURN_ON_FATAL_FAILURE(urCommandBufferExpExecutionTest::SetUp());
    for (auto &shared_ptr : shared_ptrs) {
      ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                      allocation_size, &shared_ptr));
      ASSERT_NE(shared_ptr, nullptr);
    }

    int32_t *ptrX = static_cast<int32_t *>(shared_ptrs[1]);
    int32_t *ptrY = static_cast<int32_t *>(shared_ptrs[2]);
    for (size_t i = 0; i < global_size; i++) {
      ptrX[i] = i;
      ptrY[i] = i * 2;
    }

    // Index 0 is output
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, shared_ptrs[0]));
    // Index 1 is A
    ASSERT_SUCCESS(urKernelSetArgValue(kernel, 1, sizeof(A), nullptr, &A));
    // Index 2 is X
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 2, nullptr, shared_ptrs[1]));
    // Index 3 is Y
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 3, nullptr, shared_ptrs[2]));
  }

  virtual void TearDown() override {
    for (auto &shared_ptr : shared_ptrs) {
      if (shared_ptr) {
        EXPECT_SUCCESS(urUSMFree(context, shared_ptr));
      }
    }

    UUR_RETURN_ON_FATAL_FAILURE(urCommandBufferExpExecutionTest::TearDown());
  }

  static constexpr size_t local_size = 4;
  static constexpr size_t global_size = 32;
  static constexpr size_t global_offset = 0;
  static constexpr size_t n_dimensions = 1;
  static constexpr size_t allocation_size = sizeof(uint32_t) * global_size;
  static constexpr uint32_t A = 42;
  std::array<void *, 3> shared_ptrs = {nullptr, nullptr, nullptr};
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urCommandBufferAppendKernelLaunchExpTest);
TEST_P(urCommandBufferAppendKernelLaunchExpTest, Basic) {
  ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
      cmd_buf_handle, kernel, n_dimensions, &global_offset, &global_size,
      &local_size, 0, nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr,
      nullptr));

  ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));

  ASSERT_SUCCESS(
      urEnqueueCommandBufferExp(queue, cmd_buf_handle, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  int32_t *ptrZ = static_cast<int32_t *>(shared_ptrs[0]);
  for (size_t i = 0; i < global_size; i++) {
    uint32_t result = (A * i) + (i * 2);
    ASSERT_EQ(result, ptrZ[i]);
  }
}

TEST_P(urCommandBufferAppendKernelLaunchExpTest, FinalizeTwice) {
  ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
      cmd_buf_handle, kernel, n_dimensions, &global_offset, &global_size,
      &local_size, 0, nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr,
      nullptr));

  ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));
  EXPECT_EQ_RESULT(urCommandBufferFinalizeExp(cmd_buf_handle),
                   UR_RESULT_ERROR_INVALID_OPERATION);
}

TEST_P(urCommandBufferAppendKernelLaunchExpTest, DuplicateSyncPoint) {
  ur_exp_command_buffer_sync_point_t sync_point;
  ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
      cmd_buf_handle, kernel, n_dimensions, &global_offset, &global_size,
      &local_size, 0, nullptr, 0, nullptr, 0, nullptr, &sync_point, nullptr,
      nullptr));

  // Test passing redundant sync-points
  ur_exp_command_buffer_sync_point_t sync_points[2] = {sync_point, sync_point};
  ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
      cmd_buf_handle, kernel, n_dimensions, &global_offset, &global_size,
      &local_size, 0, nullptr, 2, sync_points, 0, nullptr, nullptr, nullptr,
      nullptr));

  ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));

  ASSERT_SUCCESS(
      urEnqueueCommandBufferExp(queue, cmd_buf_handle, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));
  int32_t *ptrZ = static_cast<int32_t *>(shared_ptrs[0]);
  for (size_t i = 0; i < global_size; i++) {
    uint32_t result = (A * i) + (i * 2);
    ASSERT_EQ(result, ptrZ[i]);
  }
}
