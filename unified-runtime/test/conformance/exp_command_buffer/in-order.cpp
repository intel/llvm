// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include <array>

// Virtual base class for tests verifying that if the `isInOrder` field is
// set on command-buffer creation, then the sync-point parameters to command
// append entry-points can be omitted.
struct urInOrderCommandBufferExpTest
    : uur::command_buffer::urCommandBufferExpExecutionTest {

  virtual void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urCommandBufferExpExecutionTest::SetUp());

    ur_exp_command_buffer_desc_t desc{
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC, // stype
        nullptr,                                   // pnext
        false,                                     // isUpdatable
        true,                                      // isInOrder
        false,                                     // enableProfiling
    };
    ASSERT_SUCCESS(
        urCommandBufferCreateExp(context, device, &desc, &in_order_cb));
    ASSERT_NE(in_order_cb, nullptr);

    // Each element of Y will be initialized to its index
    std::iota(std::begin(y_data), std::end(y_data), 0);
  }

  virtual void TearDown() override {
    if (in_order_cb) {
      EXPECT_SUCCESS(urCommandBufferReleaseExp(in_order_cb));
    }

    UUR_RETURN_ON_FATAL_FAILURE(urCommandBufferExpExecutionTest::TearDown());
  }
  ur_exp_command_buffer_handle_t in_order_cb = nullptr;
  static constexpr size_t global_size = 64;
  static constexpr size_t global_offset = 0;
  static constexpr size_t allocation_size = sizeof(uint32_t) * global_size;
  static constexpr size_t n_dimensions = 1;
  static constexpr uint32_t A = 42;
  static constexpr uint32_t x_pattern = 2;
  static constexpr uint32_t zero_pattern = 0;
  std::array<uint32_t, global_size> y_data;

  void Verify(std::array<uint32_t, global_size> &output) {
    for (uint32_t i = 0; i < global_size; i++) {
      const uint32_t ref = x_pattern * A + i;
      ASSERT_EQ(ref, output[i]) << "Result mismatch at index: " << i;
    }
  }
};

struct urInOrderUSMCommandBufferExpTest : urInOrderCommandBufferExpTest {
  virtual void SetUp() override {
    program_name = "saxpy_usm";
    UUR_RETURN_ON_FATAL_FAILURE(urInOrderCommandBufferExpTest::SetUp());

    for (auto &device_ptr : device_ptrs) {
      ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                      allocation_size, &device_ptr));
      ASSERT_NE(device_ptr, nullptr);
    }

    // Index 0 is output
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, device_ptrs[0]));
    // Index 1 is A
    ASSERT_SUCCESS(urKernelSetArgValue(kernel, 1, sizeof(A), nullptr, &A));
    // Index 2 is X
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 2, nullptr, device_ptrs[1]));
    // Index 3 is Y
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 3, nullptr, device_ptrs[2]));
  }

  // Appends commands to in-order command-buffer without sync-points
  // @param[in] hints Append USM advise/prefetch hints between functional
  // commands.
  // @param[out] output Host memory to copy result back to from device pointer.
  void AppendCommands(bool hints, std::array<uint32_t, global_size> &output) {
    const uint32_t zero_pattern = 0; // Zero init the output
    ASSERT_SUCCESS(urCommandBufferAppendUSMFillExp(
        in_order_cb, device_ptrs[0], &zero_pattern, sizeof(uint32_t),
        allocation_size, 0, nullptr, 0, nullptr, nullptr, nullptr, nullptr));

    if (hints) {
      ASSERT_SUCCESS(urCommandBufferAppendUSMAdviseExp(
          in_order_cb, device_ptrs[1], allocation_size,
          UR_USM_ADVICE_FLAG_DEFAULT, 0, nullptr, 0, nullptr, nullptr, nullptr,
          nullptr));
    }

    ASSERT_SUCCESS(urCommandBufferAppendUSMFillExp(
        in_order_cb, device_ptrs[1], &x_pattern, sizeof(uint32_t),
        allocation_size, 0, nullptr, 0, nullptr, nullptr, nullptr, nullptr));

    if (hints) {
      ASSERT_SUCCESS(urCommandBufferAppendUSMPrefetchExp(
          in_order_cb, device_ptrs[0], allocation_size,
          UR_USM_MIGRATION_FLAG_DEFAULT, 0, nullptr, 0, nullptr, nullptr,
          nullptr, nullptr));
    }

    ASSERT_SUCCESS(urCommandBufferAppendUSMMemcpyExp(
        in_order_cb, device_ptrs[2], y_data.data(), allocation_size, 0, nullptr,
        0, nullptr, nullptr, nullptr, nullptr));

    if (hints) {
      ASSERT_SUCCESS(urCommandBufferAppendUSMAdviseExp(
          in_order_cb, device_ptrs[0], allocation_size,
          UR_USM_ADVICE_FLAG_DEFAULT, 0, nullptr, 0, nullptr, nullptr, nullptr,
          nullptr));
    }

    ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
        in_order_cb, kernel, n_dimensions, &global_offset, &global_size,
        nullptr, 0, nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr,
        nullptr));

    if (hints) {
      ASSERT_SUCCESS(urCommandBufferAppendUSMPrefetchExp(
          in_order_cb, device_ptrs[0], allocation_size,
          UR_USM_MIGRATION_FLAG_DEFAULT, 0, nullptr, 0, nullptr, nullptr,
          nullptr, nullptr));
    }

    ASSERT_SUCCESS(urCommandBufferAppendUSMMemcpyExp(
        in_order_cb, output.data(), device_ptrs[0], allocation_size, 0, nullptr,
        0, nullptr, nullptr, nullptr, nullptr));

    ASSERT_SUCCESS(urCommandBufferFinalizeExp(in_order_cb));
  }

  virtual void TearDown() override {
    for (auto &device_ptr : device_ptrs) {
      if (device_ptr) {
        EXPECT_SUCCESS(urUSMFree(context, device_ptr));
      }
    }
    UUR_RETURN_ON_FATAL_FAILURE(urInOrderCommandBufferExpTest::TearDown());
  }
  std::array<void *, 3> device_ptrs = {nullptr, nullptr, nullptr};
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urInOrderUSMCommandBufferExpTest);

// Tests USM Fill, Copy, and Kernel commands to a command-buffer
TEST_P(urInOrderUSMCommandBufferExpTest, WithoutHints) {
  std::array<uint32_t, global_size> output;
  AppendCommands(false, output);

  ASSERT_SUCCESS(
      urEnqueueCommandBufferExp(queue, in_order_cb, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  Verify(output);
}

// Tests USM prefetch and advise, which are hints and can be implemented by
// adapters as empty nodes, by interleaving between fill, copy, and kernel
// commands from the above test
TEST_P(urInOrderUSMCommandBufferExpTest, WithHints) {
  // No prefetch or advise in cl_khr_command_buffer
  UUR_KNOWN_FAILURE_ON(uur::OpenCL{});

  std::array<uint32_t, global_size> output;
  AppendCommands(true, output);

  ASSERT_SUCCESS(
      urEnqueueCommandBufferExp(queue, in_order_cb, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  Verify(output);
}

struct urInOrderBufferCommandBufferExpTest : urInOrderCommandBufferExpTest {
  virtual void SetUp() override {
    program_name = "saxpy";
    UUR_RETURN_ON_FATAL_FAILURE(urInOrderCommandBufferExpTest::SetUp());
    ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                     sizeof(backend), &backend, nullptr));

    for (auto &buffer : buffers) {
      ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE,
                                       allocation_size, nullptr, &buffer));

      ASSERT_NE(buffer, nullptr);
    }

    // Variable that is incremented as arguments are added to the kernel
    size_t current_arg_index = 0;
    // Index 0 is output buffer for HIP/Non-HIP
    ASSERT_SUCCESS(
        urKernelSetArgMemObj(kernel, current_arg_index++, nullptr, buffers[0]));

    // Lambda to add accessor arguments depending on backend.
    // HIP has 3 offset parameters and other backends only have 1.
    auto addAccessorArgs = [&]() {
      if (backend == UR_BACKEND_HIP) {
        size_t val = 0;
        ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_arg_index++,
                                           sizeof(size_t), nullptr, &val));
        ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_arg_index++,
                                           sizeof(size_t), nullptr, &val));
        ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_arg_index++,
                                           sizeof(size_t), nullptr, &val));
      } else {
        struct {
          size_t offsets[1] = {0};
        } accessor;
        ASSERT_SUCCESS(urKernelSetArgValue(
            kernel, current_arg_index++, sizeof(accessor), nullptr, &accessor));
      }
    };

    // Index 3 on HIP and 1 on non-HIP are accessors
    addAccessorArgs();

    // Index 4 on HIP and 2 on non-HIP is A
    ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_arg_index++, sizeof(A),
                                       nullptr, &A));

    // Index 5 on HIP and 3 on non-HIP is X buffer
    ASSERT_SUCCESS(
        urKernelSetArgMemObj(kernel, current_arg_index++, nullptr, buffers[1]));

    // Index 8 on HIP and 4 on non-HIP is X buffer accessor
    addAccessorArgs();

    // Index 9 on HIP and 5 on non-HIP is Y buffer
    ASSERT_SUCCESS(
        urKernelSetArgMemObj(kernel, current_arg_index++, nullptr, buffers[2]));

    // Index 12 on HIP and 6 on non-HIP is Y buffer accessor
    addAccessorArgs();
  }

  virtual void TearDown() override {
    for (auto &buffer : buffers) {
      if (buffer) {
        EXPECT_SUCCESS(urMemRelease(buffer));
      }
    }

    UUR_RETURN_ON_FATAL_FAILURE(urInOrderCommandBufferExpTest::TearDown());
  }

  ur_backend_t backend{};
  std::array<ur_mem_handle_t, 3> buffers = {nullptr, nullptr, nullptr};
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urInOrderBufferCommandBufferExpTest);

// Tests Buffer Fill, Write, Read, and Kernel commands to a command-buffer
TEST_P(urInOrderBufferCommandBufferExpTest, 1D) {
  // No buffer read/write command in cl_khr_command_buffer
  // See https://github.com/KhronosGroup/OpenCL-Docs/issues/1281
  UUR_KNOWN_FAILURE_ON(uur::OpenCL{});

  // Zero init the output Z
  ASSERT_SUCCESS(urCommandBufferAppendMemBufferFillExp(
      in_order_cb, buffers[0], &zero_pattern, sizeof(zero_pattern), 0,
      allocation_size, 0, nullptr, 0, nullptr, nullptr, nullptr, nullptr));

  // Initialize the X input
  ASSERT_SUCCESS(urCommandBufferAppendMemBufferFillExp(
      in_order_cb, buffers[1], &x_pattern, sizeof(x_pattern), 0,
      allocation_size, 0, nullptr, 0, nullptr, nullptr, nullptr, nullptr));

  // Initialize the Y input
  ASSERT_SUCCESS(urCommandBufferAppendMemBufferWriteExp(
      in_order_cb, buffers[2], 0, allocation_size, y_data.data(), 0, nullptr, 0,
      nullptr, nullptr, nullptr, nullptr));

  // Run kernel
  ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
      in_order_cb, kernel, n_dimensions, &global_offset, &global_size, nullptr,
      0, nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr, nullptr));

  // Copy Z -> X
  ASSERT_SUCCESS(urCommandBufferAppendMemBufferCopyExp(
      in_order_cb, buffers[0], buffers[1], 0, 0, allocation_size, 0, nullptr, 0,
      nullptr, nullptr, nullptr, nullptr));

  // Read X back to host
  std::array<uint32_t, global_size> host_data{};
  ASSERT_SUCCESS(urCommandBufferAppendMemBufferReadExp(
      in_order_cb, buffers[1], 0, allocation_size, host_data.data(), 0, nullptr,
      0, nullptr, nullptr, nullptr, nullptr));

  ASSERT_SUCCESS(urCommandBufferFinalizeExp(in_order_cb));
  ASSERT_SUCCESS(
      urEnqueueCommandBufferExp(queue, in_order_cb, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  Verify(host_data);
}

TEST_P(urInOrderBufferCommandBufferExpTest, Rect) {
  // No buffer read/write command in cl_khr_command_buffer
  // See https://github.com/KhronosGroup/OpenCL-Docs/issues/1281
  UUR_KNOWN_FAILURE_ON(uur::OpenCL{});

  // Zero init the output Z
  ASSERT_SUCCESS(urCommandBufferAppendMemBufferFillExp(
      in_order_cb, buffers[0], &zero_pattern, sizeof(zero_pattern), 0,
      allocation_size, 0, nullptr, 0, nullptr, nullptr, nullptr, nullptr));

  // Initialize the X input
  ASSERT_SUCCESS(urCommandBufferAppendMemBufferFillExp(
      in_order_cb, buffers[1], &x_pattern, sizeof(x_pattern), 0,
      allocation_size, 0, nullptr, 0, nullptr, nullptr, nullptr, nullptr));

  // Initialize the Y input
  ur_rect_offset_t origin{0, 0, 0};
  ur_rect_region_t region{16, 16, 1};
  size_t row_pitch = 16;
  size_t slice_pitch = allocation_size;
  ASSERT_SUCCESS(urCommandBufferAppendMemBufferWriteRectExp(
      in_order_cb, buffers[2], origin, origin, region, row_pitch, slice_pitch,
      row_pitch, slice_pitch, y_data.data(), 0, nullptr, 0, nullptr, nullptr,
      nullptr, nullptr));

  // Run kernel
  ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
      in_order_cb, kernel, n_dimensions, &global_offset, &global_size, nullptr,
      0, nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr, nullptr));

  // Copy Z -> X
  ASSERT_SUCCESS(urCommandBufferAppendMemBufferCopyRectExp(
      in_order_cb, buffers[0], buffers[1], origin, origin, region, row_pitch,
      slice_pitch, row_pitch, slice_pitch, 0, nullptr, 0, nullptr, nullptr,
      nullptr, nullptr));

  // Read X back to host
  std::array<uint32_t, global_size> host_data{};
  ASSERT_SUCCESS(urCommandBufferAppendMemBufferReadRectExp(
      in_order_cb, buffers[1], origin, origin, region, row_pitch, slice_pitch,
      row_pitch, slice_pitch, host_data.data(), 0, nullptr, 0, nullptr, nullptr,
      nullptr, nullptr));

  ASSERT_SUCCESS(urCommandBufferFinalizeExp(in_order_cb));
  ASSERT_SUCCESS(
      urEnqueueCommandBufferExp(queue, in_order_cb, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  Verify(host_data);
}
