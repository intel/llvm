// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

#include <cstring>

// This test runs a kernel with a mix of local memory, pointer and value args.
struct urEnqueueKernelLaunchWithArgsTest : uur::urKernelExecutionTest {
  void SetUp() override {
    program_name = "saxpy_usm_local_mem";
    UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());

    ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                     sizeof(backend), &backend, nullptr));

    // HIP has extra args for local memory so we define an offset for arg
    // indices here for updating
    hip_arg_offset = backend == UR_BACKEND_HIP ? 3 : 0;
    ur_device_usm_access_capability_flags_t shared_usm_flags;
    ASSERT_SUCCESS(
        uur::GetDeviceUSMSingleSharedSupport(device, shared_usm_flags));
    if (!(shared_usm_flags & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
      GTEST_SKIP() << "Shared USM is not supported.";
    }

    const size_t allocation_size =
        sizeof(uint32_t) * global_size[0] * local_size[0];
    for (auto &shared_ptr : shared_ptrs) {
      ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                      allocation_size, &shared_ptr));
      ASSERT_NE(shared_ptr, nullptr);

      std::vector<uint8_t> pattern(allocation_size);
      uur::generateMemFillPattern(pattern);
      std::memcpy(shared_ptr, pattern.data(), allocation_size);
    }
    uint32_t current_index = 0;
    // Index 0 is local_mem_a arg
    args.push_back({UR_STRUCTURE_TYPE_EXP_KERNEL_ARG_PROPERTIES,
                    nullptr,
                    UR_EXP_KERNEL_ARG_TYPE_LOCAL,
                    current_index++,
                    local_mem_a_size,
                    {nullptr}});

    // Hip has extra args for local mem at index 1-3
    ur_exp_kernel_arg_value_t argValue = {};
    if (backend == UR_BACKEND_HIP) {
      argValue.value = &hip_local_offset;
      ur_exp_kernel_arg_properties_t local_offset = {
          UR_STRUCTURE_TYPE_EXP_KERNEL_ARG_PROPERTIES,
          nullptr,
          UR_EXP_KERNEL_ARG_TYPE_VALUE,
          current_index++,
          sizeof(hip_local_offset),
          argValue};
      args.push_back(local_offset);
      local_offset.index = current_index++;
      args.push_back(local_offset);
      local_offset.index = current_index++;
      args.push_back(local_offset);
    }

    // Index 1 is local_mem_b arg
    args.push_back({UR_STRUCTURE_TYPE_EXP_KERNEL_ARG_PROPERTIES,
                    nullptr,
                    UR_EXP_KERNEL_ARG_TYPE_LOCAL,
                    current_index++,
                    local_mem_b_size,
                    {nullptr}});

    if (backend == UR_BACKEND_HIP) {
      argValue.value = &hip_local_offset;
      ur_exp_kernel_arg_properties_t local_offset = {
          UR_STRUCTURE_TYPE_EXP_KERNEL_ARG_PROPERTIES,
          nullptr,
          UR_EXP_KERNEL_ARG_TYPE_VALUE,
          current_index++,
          sizeof(hip_local_offset),
          argValue};
      args.push_back(local_offset);
      local_offset.index = current_index++;
      args.push_back(local_offset);
      local_offset.index = current_index++;
      args.push_back(local_offset);
    }

    // Index 2 is output
    argValue.pointer = shared_ptrs[0];
    args.push_back({UR_STRUCTURE_TYPE_EXP_KERNEL_ARG_PROPERTIES, nullptr,
                    UR_EXP_KERNEL_ARG_TYPE_POINTER, current_index++,
                    sizeof(shared_ptrs[0]), argValue});
    // Index 3 is A
    argValue.value = &A;
    args.push_back({UR_STRUCTURE_TYPE_EXP_KERNEL_ARG_PROPERTIES, nullptr,
                    UR_EXP_KERNEL_ARG_TYPE_VALUE, current_index++, sizeof(A),
                    argValue});
    // Index 4 is X
    argValue.pointer = shared_ptrs[1];
    args.push_back({UR_STRUCTURE_TYPE_EXP_KERNEL_ARG_PROPERTIES, nullptr,
                    UR_EXP_KERNEL_ARG_TYPE_POINTER, current_index++,
                    sizeof(shared_ptrs[1]), argValue});
    // Index 5 is Y
    argValue.pointer = shared_ptrs[2];
    args.push_back({UR_STRUCTURE_TYPE_EXP_KERNEL_ARG_PROPERTIES, nullptr,
                    UR_EXP_KERNEL_ARG_TYPE_POINTER, current_index++,
                    sizeof(shared_ptrs[2]), argValue});
  }

  void Validate(uint32_t *output, uint32_t *X, uint32_t *Y, uint32_t A,
                size_t length, size_t local_size) {
    for (size_t i = 0; i < length; i++) {
      uint32_t result = A * X[i] + Y[i] + local_size;
      ASSERT_EQ(result, output[i]);
    }
  }

  virtual void TearDown() override {
    for (auto &shared_ptr : shared_ptrs) {
      if (shared_ptr) {
        EXPECT_SUCCESS(urUSMFree(context, shared_ptr));
      }
    }

    UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::TearDown());
  }

  static constexpr size_t local_size[3] = {4, 1, 1};
  static constexpr size_t local_mem_a_size = local_size[0] * sizeof(uint32_t);
  static constexpr size_t local_mem_b_size = local_mem_a_size * 2;
  static constexpr size_t global_size[3] = {16, 1, 1};
  static constexpr size_t global_offset[3] = {0, 0, 0};
  static constexpr uint32_t workDim = 3;
  static constexpr uint32_t A = 42;
  std::array<void *, 5> shared_ptrs = {nullptr, nullptr, nullptr, nullptr,
                                       nullptr};

  uint32_t hip_arg_offset = 0;
  static constexpr uint64_t hip_local_offset = 0;
  ur_backend_t backend{};
  std::vector<ur_exp_kernel_arg_properties_t> args;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEnqueueKernelLaunchWithArgsTest);

TEST_P(urEnqueueKernelLaunchWithArgsTest, Success) {
  ASSERT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
      queue, kernel, workDim, global_offset, global_size, local_size,
      args.size(), args.data(), 0, nullptr, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  uint32_t *output = (uint32_t *)shared_ptrs[0];
  uint32_t *X = (uint32_t *)shared_ptrs[1];
  uint32_t *Y = (uint32_t *)shared_ptrs[2];
  Validate(output, X, Y, A, global_size[0], local_size[0]);
}

TEST_P(urEnqueueKernelLaunchWithArgsTest, InvalidNullHandleQueue) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urEnqueueKernelLaunchWithArgsExp(
                       nullptr, kernel, workDim, global_offset, global_size,
                       local_size, args.size(), args.data(), 0, nullptr, 0,
                       nullptr, nullptr));
}

TEST_P(urEnqueueKernelLaunchWithArgsTest, InvalidNullHandleKernel) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urEnqueueKernelLaunchWithArgsExp(
                       queue, nullptr, workDim, global_offset, global_size,
                       local_size, args.size(), args.data(), 0, nullptr, 0,
                       nullptr, nullptr));
}

TEST_P(urEnqueueKernelLaunchWithArgsTest, InvalidNullPointerGlobalSize) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urEnqueueKernelLaunchWithArgsExp(
                       queue, kernel, workDim, global_offset, nullptr,
                       local_size, args.size(), args.data(), 0, nullptr, 0,
                       nullptr, nullptr));
}

TEST_P(urEnqueueKernelLaunchWithArgsTest, InvalidNullPointerProperties) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urEnqueueKernelLaunchWithArgsExp(
                       queue, kernel, workDim, global_offset, global_size,
                       local_size, args.size(), args.data(), 1, nullptr, 0,
                       nullptr, nullptr));
}

TEST_P(urEnqueueKernelLaunchWithArgsTest, InvalidNullPointerArgs) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urEnqueueKernelLaunchWithArgsExp(
                       queue, kernel, workDim, global_offset, global_size,
                       local_size, args.size(), nullptr, 0, nullptr, 0, nullptr,
                       nullptr));
}

TEST_P(urEnqueueKernelLaunchWithArgsTest, InvalidEventWaitList) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
                   urEnqueueKernelLaunchWithArgsExp(
                       queue, kernel, workDim, global_offset, global_size,
                       local_size, args.size(), args.data(), 0, nullptr, 1,
                       nullptr, nullptr));
  ur_event_handle_t event = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
                   urEnqueueKernelLaunchWithArgsExp(
                       queue, kernel, workDim, global_offset, global_size,
                       local_size, args.size(), args.data(), 0, nullptr, 0,
                       &event, nullptr));
}

// This test runs a kernel with a buffer (MEM_OBJ) arg.
struct urEnqueueKernelLaunchWithArgsMemObjTest : uur::urKernelExecutionTest {
  void SetUp() override {
    program_name = "fill";
    UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());

    ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                     sizeof(backend), &backend, nullptr));

    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE,
                                     sizeof(val) * global_size[0], nullptr,
                                     &buffer));

    char zero = 0;
    ASSERT_SUCCESS(urEnqueueMemBufferFill(queue, buffer, &zero, sizeof(zero), 0,
                                          buffer_size, 0, nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));

    // First argument is buffer to fill
    unsigned current_arg_index = 0;
    ur_exp_kernel_arg_mem_obj_tuple_t buffer_and_properties = {buffer, 0};
    ur_exp_kernel_arg_properties_t arg = {
        UR_STRUCTURE_TYPE_EXP_KERNEL_ARG_PROPERTIES,
        nullptr,
        UR_EXP_KERNEL_ARG_TYPE_MEM_OBJ,
        current_arg_index++,
        sizeof(buffer),
        {nullptr}};
    arg.value.memObjTuple = buffer_and_properties;
    args.push_back(arg);

    // Add accessor arguments depending on backend.
    // HIP has 3 offset parameters and other backends only have 1.
    if (backend == UR_BACKEND_HIP) {
      arg.type = UR_EXP_KERNEL_ARG_TYPE_VALUE;
      arg.size = sizeof(hip_local_offset);
      arg.value.value = &hip_local_offset;
      arg.index = current_arg_index++;
      args.push_back(arg);
      arg.index = current_arg_index++;
      args.push_back(arg);
      arg.index = current_arg_index++;
      args.push_back(arg);
    } else {
      arg.type = UR_EXP_KERNEL_ARG_TYPE_VALUE;
      arg.index = current_arg_index++;
      arg.size = sizeof(accessor);
      arg.value.value = &accessor;
      args.push_back(arg);
    }

    // Second user defined argument is scalar to fill with.
    arg.type = UR_EXP_KERNEL_ARG_TYPE_VALUE;
    arg.index = current_arg_index++;
    arg.size = sizeof(val);
    arg.value.value = &val;
    args.push_back(arg);
  }

  void TearDown() override {
    if (buffer) {
      EXPECT_SUCCESS(urMemRelease(buffer));
    }

    UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::TearDown());
  }

  static constexpr uint32_t val = 42;
  static constexpr size_t global_size[3] = {32, 1, 1};
  static constexpr uint32_t workDim = 3;
  static constexpr size_t buffer_size = sizeof(val) * global_size[0];
  static constexpr uint64_t hip_local_offset = 0;
  ur_backend_t backend{};
  ur_mem_handle_t buffer = nullptr;
  // This is the accessor offset struct sycl kernels expect to accompany buffer args.
  struct {
    size_t offsets[1] = {0};
  } accessor;
  std::vector<ur_exp_kernel_arg_properties_t> args;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEnqueueKernelLaunchWithArgsMemObjTest);

TEST_P(urEnqueueKernelLaunchWithArgsMemObjTest, Success) {
  ASSERT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
      queue, kernel, workDim, nullptr, global_size, nullptr, args.size(),
      args.data(), 0, nullptr, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));
  ValidateBuffer(buffer, buffer_size, val);
}
