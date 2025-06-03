// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

#include <cstring>

// Test launching kernels with multiple local arguments return the expected
// outputs
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
    if (backend == UR_BACKEND_HIP) {
      ur_exp_kernel_arg_properties_t local_offset = {
          UR_STRUCTURE_TYPE_EXP_KERNEL_ARG_PROPERTIES,
          nullptr,
          UR_EXP_KERNEL_ARG_TYPE_VALUE,
          current_index++,
          sizeof(hip_local_offset),
          {&hip_local_offset}};
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
      ur_exp_kernel_arg_properties_t local_offset = {
          UR_STRUCTURE_TYPE_EXP_KERNEL_ARG_PROPERTIES,
          nullptr,
          UR_EXP_KERNEL_ARG_TYPE_VALUE,
          current_index++,
          sizeof(hip_local_offset),
          {&hip_local_offset}};
      args.push_back(local_offset);
      local_offset.index = current_index++;
      args.push_back(local_offset);
      local_offset.index = current_index++;
      args.push_back(local_offset);
    }

    // Index 2 is output
    args.push_back({UR_STRUCTURE_TYPE_EXP_KERNEL_ARG_PROPERTIES,
                    nullptr,
                    UR_EXP_KERNEL_ARG_TYPE_POINTER,
                    current_index++,
                    sizeof(shared_ptrs[0]),
                    {shared_ptrs[0]}});
    // Index 3 is A
    args.push_back({UR_STRUCTURE_TYPE_EXP_KERNEL_ARG_PROPERTIES,
                    nullptr,
                    UR_EXP_KERNEL_ARG_TYPE_VALUE,
                    current_index++,
                    sizeof(A),
                    {(void *)&A}});
    // Index 4 is X
    args.push_back({UR_STRUCTURE_TYPE_EXP_KERNEL_ARG_PROPERTIES,
                    nullptr,
                    UR_EXP_KERNEL_ARG_TYPE_POINTER,
                    current_index++,
                    sizeof(shared_ptrs[1]),
                    {shared_ptrs[1]}});
    // Index 5 is Y
    args.push_back({UR_STRUCTURE_TYPE_EXP_KERNEL_ARG_PROPERTIES,
                    nullptr,
                    UR_EXP_KERNEL_ARG_TYPE_POINTER,
                    current_index++,
                    sizeof(shared_ptrs[2]),
                    {shared_ptrs[2]}});
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
      queue, kernel, global_offset, global_size, local_size, args.size(),
      args.data(), 0, nullptr, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  uint32_t *output = (uint32_t *)shared_ptrs[0];
  uint32_t *X = (uint32_t *)shared_ptrs[1];
  uint32_t *Y = (uint32_t *)shared_ptrs[2];
  Validate(output, X, Y, A, global_size[0], local_size[0]);
}
