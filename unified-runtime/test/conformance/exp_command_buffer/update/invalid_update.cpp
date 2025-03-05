// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../fixtures.h"
#include <array>
#include <cstring>

// Negative tests that correct error codes are thrown on invalid update usage.
struct InvalidUpdateTest
    : uur::command_buffer::urUpdatableCommandBufferExpExecutionTest {
  void SetUp() override {
    program_name = "fill_usm";
    UUR_RETURN_ON_FATAL_FAILURE(
        urUpdatableCommandBufferExpExecutionTest::SetUp());

    ur_device_usm_access_capability_flags_t shared_usm_flags;
    ASSERT_SUCCESS(
        uur::GetDeviceUSMSingleSharedSupport(device, shared_usm_flags));
    if (!(shared_usm_flags & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
      GTEST_SKIP() << "Shared USM is not supported.";
    }

    // Allocate USM pointer to fill
    ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                    allocation_size, &shared_ptr));
    ASSERT_NE(shared_ptr, nullptr);
    std::memset(shared_ptr, 0, allocation_size);

    // Index 0 is output
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, shared_ptr));
    // Index 1 is input scalar
    ASSERT_SUCCESS(urKernelSetArgValue(kernel, 1, sizeof(val), nullptr, &val));

    // Append kernel command to command-buffer
    ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
        updatable_cmd_buf_handle, kernel, n_dimensions, &global_offset,
        &global_size, &local_size, 0, nullptr, 0, nullptr, 0, nullptr, nullptr,
        nullptr, &command_handle));
    ASSERT_NE(command_handle, nullptr);
  }

  void TearDown() override {
    // Workaround an issue with the OpenCL adapter implementing urUsmFree
    // using a blocking free where hangs
    if (updatable_cmd_buf_handle && !finalized) {
      EXPECT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));
    }

    if (shared_ptr) {
      EXPECT_SUCCESS(urUSMFree(context, shared_ptr));
    }

    UUR_RETURN_ON_FATAL_FAILURE(
        urUpdatableCommandBufferExpExecutionTest::TearDown());
  }

  static constexpr uint32_t val = 42;
  static constexpr size_t local_size = 4;
  static constexpr size_t global_size = 32;
  static constexpr size_t global_offset = 0;
  static constexpr uint32_t n_dimensions = 1;
  static constexpr size_t allocation_size = sizeof(val) * global_size;
  void *shared_ptr = nullptr;
  ur_exp_command_buffer_command_handle_t command_handle = nullptr;
  bool finalized = false;
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(InvalidUpdateTest);

// Test error code is returned if command-buffer not finalized
TEST_P(InvalidUpdateTest, NotFinalizedCommandBuffer) {
  // Set new value to use for fill at kernel index 1
  uint32_t new_val = 33;
  ur_exp_command_buffer_update_value_arg_desc_t new_input_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      1,                                                          // argIndex
      sizeof(new_val),                                            // argSize
      nullptr,                                                    // pProperties
      &new_val,                                                   // hArgValue
  };

  ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
      nullptr,                                                        // pNext
      command_handle,  // hCommand
      kernel,          // hNewKernel
      0,               // numNewMemObjArgs
      0,               // numNewPointerArgs
      1,               // numNewValueArgs
      n_dimensions,    // newWorkDim
      nullptr,         // pNewMemObjArgList
      nullptr,         // pNewPointerArgList
      &new_input_desc, // pNewValueArgList
      nullptr,         // pNewGlobalWorkOffset
      nullptr,         // pNewGlobalWorkSize
      nullptr,         // pNewLocalWorkSize
  };

  // Update command to command-buffer that has not been finalized
  ur_result_t result = urCommandBufferUpdateKernelLaunchExp(
      updatable_cmd_buf_handle, 1, &update_desc);
  ASSERT_EQ(UR_RESULT_ERROR_INVALID_OPERATION, result);
}

// Test error code is returned if command-buffer not created with isUpdatable
TEST_P(InvalidUpdateTest, NotUpdatableCommandBuffer) {
  // Create a command-buffer without isUpdatable
  ur_exp_command_buffer_handle_t test_cmd_buf_handle = nullptr;
  ur_exp_command_buffer_desc_t desc{
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC, nullptr, false, false, false,
  };
  ASSERT_SUCCESS(
      urCommandBufferCreateExp(context, device, &desc, &test_cmd_buf_handle));
  EXPECT_NE(test_cmd_buf_handle, nullptr);

  // Append a kernel commands to command-buffer and close command-buffer
  // Should be an error because we are trying to get command handle but
  // command-buffer is not updatable.
  ur_exp_command_buffer_command_handle_t test_command_handle = nullptr;
  ASSERT_EQ_RESULT(urCommandBufferAppendKernelLaunchExp(
                       test_cmd_buf_handle, kernel, n_dimensions,
                       &global_offset, &global_size, &local_size, 0, nullptr, 0,
                       nullptr, 0, nullptr, nullptr, nullptr,
                       &test_command_handle),
                   UR_RESULT_ERROR_INVALID_OPERATION);
  ASSERT_EQ(test_command_handle, nullptr);

  EXPECT_SUCCESS(urCommandBufferFinalizeExp(test_cmd_buf_handle));
  finalized = true;

  // Set new value to use for fill at kernel index 1
  uint32_t new_val = 33;
  ur_exp_command_buffer_update_value_arg_desc_t new_input_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      1,                                                          // argIndex
      sizeof(new_val),                                            // argSize
      nullptr,                                                    // pProperties
      &new_val,                                                   // hArgValue
  };

  ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
      nullptr,                                                        // pNext
      test_command_handle, // hCommand
      kernel,              // hNewKernel
      0,                   // numNewMemObjArgs
      0,                   // numNewPointerArgs
      1,                   // numNewValueArgs
      n_dimensions,        // newWorkDim
      nullptr,             // pNewMemObjArgList
      nullptr,             // pNewPointerArgList
      &new_input_desc,     // pNewValueArgList
      nullptr,             // pNewGlobalWorkOffset
      nullptr,             // pNewGlobalWorkSize
      nullptr,             // pNewLocalWorkSize
  };

  // Since no command handle was returned Update command to command-buffer
  // should also be an error.
  ur_result_t result = urCommandBufferUpdateKernelLaunchExp(test_cmd_buf_handle,
                                                            1, &update_desc);
  EXPECT_EQ(UR_RESULT_ERROR_INVALID_NULL_HANDLE, result);

  if (test_cmd_buf_handle) {
    EXPECT_SUCCESS(urCommandBufferReleaseExp(test_cmd_buf_handle));
  }
}

// If the number of dimensions change, then the global work size and offset
// also need to be updated.
TEST_P(InvalidUpdateTest, InvalidDimensions) {
  ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));
  finalized = true;

  uint32_t new_dimensions = 2;
  std::array<size_t, 2> new_global_offset{0, 0};
  std::array<size_t, 2> new_global_size{64, 64};

  ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
      nullptr,                                                        // pNext
      command_handle,         // hCommand
      kernel,                 // hNewKernel
      0,                      // numNewMemObjArgs
      0,                      // numNewPointerArgs
      0,                      // numNewValueArgs
      new_dimensions,         // newWorkDim
      nullptr,                // pNewMemObjArgList
      nullptr,                // pNewPointerArgList
      nullptr,                // pNewValueArgList
      nullptr,                // pNewGlobalWorkOffset
      new_global_size.data(), // pNewGlobalWorkSize
      nullptr,                // pNewLocalWorkSize
  };

  ASSERT_EQ(UR_RESULT_ERROR_INVALID_VALUE,
            urCommandBufferUpdateKernelLaunchExp(updatable_cmd_buf_handle, 1,
                                                 &update_desc));

  update_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
      nullptr,                                                        // pNext
      command_handle,           // hCommand
      kernel,                   // hNewKernel
      0,                        // numNewMemObjArgs
      0,                        // numNewPointerArgs
      0,                        // numNewValueArgs
      new_dimensions,           // newWorkDim
      nullptr,                  // pNewMemObjArgList
      nullptr,                  // pNewPointerArgList
      nullptr,                  // pNewValueArgList
      new_global_offset.data(), // pNewGlobalWorkOffset
      nullptr,                  // pNewGlobalWorkSize
      nullptr,                  // pNewLocalWorkSize
  };

  ASSERT_EQ(UR_RESULT_ERROR_INVALID_VALUE,
            urCommandBufferUpdateKernelLaunchExp(updatable_cmd_buf_handle, 1,
                                                 &update_desc));
}

// If the command-handle isn't valid an error should be returned
TEST_P(InvalidUpdateTest, InvalidCommandHandle) {
  ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));
  finalized = true;

  ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
      nullptr,                                                        // pNext
      nullptr,      // hCommand
      kernel,       // hNewKernel
      0,            // numNewMemObjArgs
      0,            // numNewPointerArgs
      0,            // numNewValueArgs
      n_dimensions, // newWorkDim
      nullptr,      // pNewMemObjArgList
      nullptr,      // pNewPointerArgList
      nullptr,      // pNewValueArgList
      nullptr,      // pNewGlobalWorkOffset
      nullptr,      // pNewGlobalWorkSize
      nullptr,      // pNewLocalWorkSize
  };

  ASSERT_EQ(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
            urCommandBufferUpdateKernelLaunchExp(updatable_cmd_buf_handle, 1,
                                                 &update_desc));
}

// Test error code is returned if command handle and command-buffer is
// mismatched
TEST_P(InvalidUpdateTest, CommandBufferMismatch) {
  // Create a command-buffer with update enabled.
  ur_exp_command_buffer_handle_t test_cmd_buf_handle = nullptr;
  ur_exp_command_buffer_desc_t desc{UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC,
                                    nullptr, true, false, false};
  ASSERT_SUCCESS(
      urCommandBufferCreateExp(context, device, &desc, &test_cmd_buf_handle));
  EXPECT_NE(test_cmd_buf_handle, nullptr);

  EXPECT_SUCCESS(urCommandBufferFinalizeExp(test_cmd_buf_handle));
  EXPECT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));
  finalized = true;

  // Set new value to use for fill at kernel index 1
  uint32_t new_val = 33;
  ur_exp_command_buffer_update_value_arg_desc_t new_input_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      1,                                                          // argIndex
      sizeof(new_val),                                            // argSize
      nullptr,                                                    // pProperties
      &new_val,                                                   // hArgValue
  };

  ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
      nullptr,                                                        // pNext
      command_handle,  // hCommand
      kernel,          // hNewKernel
      0,               // numNewMemObjArgs
      0,               // numNewPointerArgs
      1,               // numNewValueArgs
      n_dimensions,    // newWorkDim
      nullptr,         // pNewMemObjArgList
      nullptr,         // pNewPointerArgList
      &new_input_desc, // pNewValueArgList
      nullptr,         // pNewGlobalWorkOffset
      nullptr,         // pNewGlobalWorkSize
      nullptr,         // pNewLocalWorkSize
  };

  // Since no command handle was returned Update command to command-buffer
  // should also be an error.
  ur_result_t result = urCommandBufferUpdateKernelLaunchExp(test_cmd_buf_handle,
                                                            1, &update_desc);
  EXPECT_EQ(UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_COMMAND_HANDLE_EXP, result);

  if (test_cmd_buf_handle) {
    EXPECT_SUCCESS(urCommandBufferReleaseExp(test_cmd_buf_handle));
  }
}

// Tests that an error is thrown when trying to update a kernel capability
// that isn't supported.
struct InvalidUpdateCommandBufferExpExecutionTest : uur::urKernelExecutionTest {
  void SetUp() override {
    program_name = "fill_usm";
    UUR_RETURN_ON_FATAL_FAILURE(uur::urKernelExecutionTest::SetUp());

    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::checkCommandBufferSupport(device));

    ASSERT_SUCCESS(urDeviceGetInfo(
        device, UR_DEVICE_INFO_COMMAND_BUFFER_UPDATE_CAPABILITIES_EXP,
        sizeof(update_capability_flags), &update_capability_flags, nullptr));

    if (0 == update_capability_flags) {
      GTEST_SKIP() << "Test requires update support from device";
    }

    ur_device_usm_access_capability_flags_t shared_usm_flags;
    ASSERT_SUCCESS(
        uur::GetDeviceUSMSingleSharedSupport(device, shared_usm_flags));
    if (!(shared_usm_flags & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
      GTEST_SKIP() << "Shared USM is not supported.";
    }

    // Allocate USM pointer to fill
    ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                    allocation_size, &shared_ptr));
    ASSERT_NE(shared_ptr, nullptr);
    std::memset(shared_ptr, 0, allocation_size);

    // Index 0 is output
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, shared_ptr));
    // Index 1 is input scalar
    ASSERT_SUCCESS(urKernelSetArgValue(kernel, 1, sizeof(val), nullptr, &val));

    // Create a command-buffer with update enabled.
    ur_exp_command_buffer_desc_t desc{UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC,
                                      nullptr, true, false, false};

    ASSERT_SUCCESS(urCommandBufferCreateExp(context, device, &desc,
                                            &updatable_cmd_buf_handle));
    ASSERT_NE(updatable_cmd_buf_handle, nullptr);

    // Append kernel command to command-buffer
    ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
        updatable_cmd_buf_handle, kernel, n_dimensions, &global_offset,
        &global_size, &local_size, 0, nullptr, 0, nullptr, 0, nullptr, nullptr,
        nullptr, &command_handle));
    ASSERT_NE(command_handle, nullptr);

    ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));

    ASSERT_SUCCESS(urKernelCreate(program, kernel_name.data(), &kernel_2));
  }

  void TearDown() override {
    if (updatable_cmd_buf_handle) {
      EXPECT_SUCCESS(urCommandBufferReleaseExp(updatable_cmd_buf_handle));
    }

    if (shared_ptr) {
      EXPECT_SUCCESS(urUSMFree(context, shared_ptr));
    }

    if (kernel_2) {
      ASSERT_SUCCESS(urKernelRelease(kernel_2));
    }

    UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::TearDown());
  }

  ur_exp_command_buffer_handle_t updatable_cmd_buf_handle = nullptr;
  ur_exp_command_buffer_command_handle_t command_handle = nullptr;
  ur_kernel_handle_t kernel_2 = nullptr;
  static constexpr uint32_t val = 42;
  static constexpr size_t local_size = 4;
  static constexpr size_t global_size = 32;
  static constexpr size_t global_offset = 0;
  static constexpr uint32_t n_dimensions = 1;
  static constexpr size_t allocation_size = sizeof(val) * global_size;
  void *shared_ptr = nullptr;
  ur_device_command_buffer_update_capability_flags_t update_capability_flags =
      0;
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(InvalidUpdateCommandBufferExpExecutionTest);

// Test error reported if device doesn't support updating kernel args
TEST_P(InvalidUpdateCommandBufferExpExecutionTest, KernelArg) {
  if (update_capability_flags &
      UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_KERNEL_ARGUMENTS) {
    GTEST_SKIP() << "Test requires device to not support kernel arg update "
                    "capability";
  }

  uint32_t new_val = 33;
  ur_exp_command_buffer_update_value_arg_desc_t new_input_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
      nullptr,                                                    // pNext
      1,                                                          // argIndex
      sizeof(new_val),                                            // argSize
      nullptr,                                                    // pProperties
      &new_val,                                                   // hArgValue
  };

  ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
      nullptr,                                                        // pNext
      command_handle,  // hCommand
      nullptr,         // hNewKernel
      0,               // numNewMemObjArgs
      0,               // numNewPointerArgs
      1,               // numNewValueArgs
      n_dimensions,    // newWorkDim
      nullptr,         // pNewMemObjArgList
      nullptr,         // pNewPointerArgList
      &new_input_desc, // pNewValueArgList
      nullptr,         // pNewGlobalWorkOffset
      nullptr,         // pNewGlobalWorkSize
      nullptr,         // pNewLocalWorkSize
  };

  ur_result_t result = urCommandBufferUpdateKernelLaunchExp(
      updatable_cmd_buf_handle, 1, &update_desc);
  ASSERT_EQ(UR_RESULT_ERROR_UNSUPPORTED_FEATURE, result);
}

TEST_P(InvalidUpdateCommandBufferExpExecutionTest, GlobalSize) {
  if (update_capability_flags &
      UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_GLOBAL_WORK_SIZE) {
    GTEST_SKIP() << "Test requires device to not support global work size "
                    "update capability.";
  }

  auto new_global_size = global_size * 2;
  ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
      nullptr,                                                        // pNext
      command_handle,   // hCommand
      nullptr,          // hNewKernel
      0,                // numNewMemObjArgs
      0,                // numNewPointerArgs
      0,                // numNewValueArgs
      n_dimensions,     // newWorkDim
      nullptr,          // pNewMemObjArgList
      nullptr,          // pNewPointerArgList
      nullptr,          // pNewValueArgList
      nullptr,          // pNewGlobalWorkOffset
      &new_global_size, // pNewGlobalWorkSize
      nullptr,          // pNewLocalWorkSize
  };

  ur_result_t result = urCommandBufferUpdateKernelLaunchExp(
      updatable_cmd_buf_handle, 1, &update_desc);
  ASSERT_EQ(UR_RESULT_ERROR_UNSUPPORTED_FEATURE, result);
}

TEST_P(InvalidUpdateCommandBufferExpExecutionTest, GlobalOffset) {
  if (update_capability_flags &
      UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_GLOBAL_WORK_OFFSET) {
    GTEST_SKIP() << "Test requires device to not support global work "
                    "offset update capability.";
  }

  size_t new_global_offset = 1;
  ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
      nullptr,                                                        // pNext
      command_handle,     // hCommand
      nullptr,            // hNewKernel
      0,                  // numNewMemObjArgs
      0,                  // numNewPointerArgs
      0,                  // numNewValueArgs
      n_dimensions,       // newWorkDim
      nullptr,            // pNewMemObjArgList
      nullptr,            // pNewPointerArgList
      nullptr,            // pNewValueArgList
      &new_global_offset, // pNewGlobalWorkOffset
      nullptr,            // pNewGlobalWorkSize
      nullptr,            // pNewLocalWorkSize
  };

  ur_result_t result = urCommandBufferUpdateKernelLaunchExp(
      updatable_cmd_buf_handle, 1, &update_desc);
  ASSERT_EQ(UR_RESULT_ERROR_UNSUPPORTED_FEATURE, result);
}

TEST_P(InvalidUpdateCommandBufferExpExecutionTest, LocalSize) {
  if (update_capability_flags &
      UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_LOCAL_WORK_SIZE) {
    GTEST_SKIP() << "Test requires device to not support local work size "
                    "update capability.";
  }

  size_t new_local_size = 2;
  ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
      nullptr,                                                        // pNext
      command_handle,  // hCommand
      nullptr,         // hNewKernel
      0,               // numNewMemObjArgs
      0,               // numNewPointerArgs
      0,               // numNewValueArgs
      n_dimensions,    // newWorkDim
      nullptr,         // pNewMemObjArgList
      nullptr,         // pNewPointerArgList
      nullptr,         // pNewValueArgList
      nullptr,         // pNewGlobalWorkOffset
      nullptr,         // pNewGlobalWorkSize
      &new_local_size, // pNewLocalWorkSize
  };

  ur_result_t result = urCommandBufferUpdateKernelLaunchExp(
      updatable_cmd_buf_handle, 1, &update_desc);
  ASSERT_EQ(UR_RESULT_ERROR_UNSUPPORTED_FEATURE, result);
}

TEST_P(InvalidUpdateCommandBufferExpExecutionTest, ImplChosenLocalSize) {
  bool local_update_support =
      update_capability_flags &
      UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_LOCAL_WORK_SIZE;
  bool global_update_support =
      update_capability_flags &
      UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_GLOBAL_WORK_SIZE;

  if (local_update_support || !global_update_support) {
    GTEST_SKIP() << "Test requires device to not support local work size "
                    "update capability, but support global work size update.";
  }

  auto new_global_size = global_size * 2;
  ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
      nullptr,                                                        // pNext
      command_handle,   // hCommand
      nullptr,          // hNewKernel
      0,                // numNewMemObjArgs
      0,                // numNewPointerArgs
      0,                // numNewValueArgs
      n_dimensions,     // newWorkDim
      nullptr,          // pNewMemObjArgList
      nullptr,          // pNewPointerArgList
      nullptr,          // pNewValueArgList
      nullptr,          // pNewGlobalWorkOffset
      &new_global_size, // pNewGlobalWorkSize
      nullptr,          // pNewLocalWorkSize
  };

  ur_result_t result = urCommandBufferUpdateKernelLaunchExp(
      updatable_cmd_buf_handle, 1, &update_desc);
  ASSERT_EQ(UR_RESULT_ERROR_UNSUPPORTED_FEATURE, result);
}

TEST_P(InvalidUpdateCommandBufferExpExecutionTest, Kernel) {
  if (update_capability_flags &
      UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_KERNEL_HANDLE) {
    GTEST_SKIP() << "Test requires device to not support kernel handle "
                    "update capability.";
  }

  ur_exp_command_buffer_update_kernel_launch_desc_t update_desc = {
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
      nullptr,                                                        // pNext
      command_handle, // hCommand
      kernel_2,       // hNewKernel
      0,              // numNewMemObjArgs
      0,              // numNewPointerArgs
      0,              // numNewValueArgs
      n_dimensions,   // newWorkDim
      nullptr,        // pNewMemObjArgList
      nullptr,        // pNewPointerArgList
      nullptr,        // pNewValueArgList
      nullptr,        // pNewGlobalWorkOffset
      nullptr,        // pNewGlobalWorkSize
      nullptr,        // pNewLocalWorkSize
  };

  ur_result_t result = urCommandBufferUpdateKernelLaunchExp(
      updatable_cmd_buf_handle, 1, &update_desc);
  ASSERT_EQ(UR_RESULT_ERROR_UNSUPPORTED_FEATURE, result);
}
