// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../fixtures.h"
#include "uur/raii.h"
#include <array>
#include <cstring>

struct TestSaxpyKernel : public uur::command_buffer::TestKernel {

  TestSaxpyKernel(ur_platform_handle_t Platform, ur_context_handle_t Context,
                  ur_device_handle_t Device)
      : TestKernel("saxpy_usm", Platform, Context, Device) {}

  ~TestSaxpyKernel() override = default;

  void setUpKernel() override {

    ASSERT_NO_FATAL_FAILURE(buildKernel());

    const size_t AllocationSize = sizeof(uint32_t) * GlobalSize;
    for (auto &SharedPtr : Allocations) {
      ASSERT_SUCCESS(urUSMSharedAlloc(Context, Device, nullptr, nullptr,
                                      AllocationSize, &SharedPtr));
      ASSERT_NE(SharedPtr, nullptr);

      std::vector<uint8_t> pattern(AllocationSize);
      uur::generateMemFillPattern(pattern);
      std::memcpy(SharedPtr, pattern.data(), AllocationSize);
    }

    // Index 0 is the output
    ASSERT_SUCCESS(urKernelSetArgPointer(Kernel, 0, nullptr, Allocations[0]));
    // Index 1 is A
    ASSERT_SUCCESS(urKernelSetArgValue(Kernel, 1, sizeof(A), nullptr, &A));
    // Index 2 is X
    ASSERT_SUCCESS(urKernelSetArgPointer(Kernel, 2, nullptr, Allocations[1]));
    // Index 3 is Y
    ASSERT_SUCCESS(urKernelSetArgPointer(Kernel, 3, nullptr, Allocations[2]));

    UpdatePointerDesc[0] = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
        nullptr,                                                      // pNext
        2,               // argIndex
        nullptr,         // pProperties
        &Allocations[0], // pArgValue
    };

    UpdatePointerDesc[1] = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
        nullptr,                                                      // pNext
        2,               // argIndex
        nullptr,         // pProperties
        &Allocations[1], // pArgValue
    };

    UpdatePointerDesc[2] = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
        nullptr,                                                      // pNext
        3,               // argIndex
        nullptr,         // pProperties
        &Allocations[2], // pArgValue
    };

    UpdateValDesc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        1,                                                          // argIndex
        sizeof(A),                                                  // argSize
        nullptr, // pProperties
        &A,      // hArgValue
    };

    UpdateDesc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
        nullptr,                                                        // pNext
        nullptr,                  // hCommand
        Kernel,                   // hNewKernel
        0,                        // numNewMemObjArgs
        3,                        // numNewPointerArgs
        1,                        // numNewValueArgs
        NDimensions,              // newWorkDim
        nullptr,                  // pNewMemObjArgList
        UpdatePointerDesc.data(), // pNewPointerArgList
        &UpdateValDesc,           // pNewValueArgList
        &GlobalOffset,            // pNewGlobalWorkOffset
        &GlobalSize,              // pNewGlobalWorkSize
        &LocalSize,               // pNewLocalWorkSize
    };
  }

  void destroyKernel() override {
    for (auto &Allocation : Allocations) {
      if (Allocation) {
        EXPECT_SUCCESS(urUSMFree(Context, Allocation));
      }
    }
    ASSERT_NO_FATAL_FAILURE(TestKernel::destroyKernel());
  }

  void validate() override {
    auto *output = static_cast<uint32_t *>(Allocations[0]);
    auto *X = static_cast<uint32_t *>(Allocations[1]);
    auto *Y = static_cast<uint32_t *>(Allocations[2]);

    for (size_t i = 0; i < GlobalSize; i++) {
      uint32_t result = A * X[i] + Y[i];
      ASSERT_EQ(result, output[i]);
    }
  }

  std::array<ur_exp_command_buffer_update_pointer_arg_desc_t, 3>
      UpdatePointerDesc;
  ur_exp_command_buffer_update_value_arg_desc_t UpdateValDesc;
  ur_exp_command_buffer_update_kernel_launch_desc_t UpdateDesc;

  size_t LocalSize = 4;
  size_t GlobalSize = 32;
  size_t GlobalOffset = 0;
  uint32_t NDimensions = 1;
  uint32_t A = 42;

  std::array<void *, 3> Allocations = {nullptr, nullptr, nullptr};
};

struct TestFill2DKernel : public uur::command_buffer::TestKernel {

  TestFill2DKernel(ur_platform_handle_t Platform, ur_context_handle_t Context,
                   ur_device_handle_t Device)
      : TestKernel("fill_usm_2d", Platform, Context, Device) {}

  ~TestFill2DKernel() override = default;

  void setUpKernel() override {
    ASSERT_NO_FATAL_FAILURE(buildKernel());

    const size_t allocation_size = sizeof(uint32_t) * SizeX * SizeY;
    ASSERT_SUCCESS(urUSMSharedAlloc(Context, Device, nullptr, nullptr,
                                    allocation_size, &Memory));

    // Index 0 is the output
    ASSERT_SUCCESS(urKernelSetArgPointer(Kernel, 0, nullptr, Memory));
    // Index 1 is the fill value
    ASSERT_SUCCESS(urKernelSetArgValue(Kernel, 1, sizeof(Val), nullptr, &Val));

    ASSERT_NE(Memory, nullptr);

    std::vector<uint8_t> pattern(allocation_size);
    uur::generateMemFillPattern(pattern);
    std::memcpy(Memory, pattern.data(), allocation_size);

    UpdatePointerDesc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC, // stype
        nullptr,                                                      // pNext
        0,       // argIndex
        nullptr, // pProperties
        &Memory, // pArgValue
    };

    UpdateValDesc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC, // stype
        nullptr,                                                    // pNext
        1,                                                          // argIndex
        sizeof(Val),                                                // argSize
        nullptr, // pProperties
        &Val,    // hArgValue
    };

    UpdateDesc = {
        UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_KERNEL_LAUNCH_DESC, // stype
        nullptr,                                                        // pNext
        nullptr,             // hCommand
        Kernel,              // hNewKernel
        0,                   // numNewMemObjArgs
        1,                   // numNewPointerArgs
        1,                   // numNewValueArgs
        NDimensions,         // newWorkDim
        nullptr,             // pNewMemObjArgList
        &UpdatePointerDesc,  // pNewPointerArgList
        &UpdateValDesc,      // pNewValueArgList
        GlobalOffset.data(), // pNewGlobalWorkOffset
        GlobalSize.data(),   // pNewGlobalWorkSize
        LocalSize.data(),    // pNewLocalWorkSize
    };
  }

  void destroyKernel() override {
    if (Memory) {
      EXPECT_SUCCESS(urUSMFree(Context, Memory));
    }
    ASSERT_NO_FATAL_FAILURE(TestKernel::destroyKernel());
  }

  void validate() override {
    for (size_t i = 0; i < SizeX * SizeY; i++) {
      ASSERT_EQ(static_cast<uint32_t *>(Memory)[i], Val);
    }
  }

  ur_exp_command_buffer_update_pointer_arg_desc_t UpdatePointerDesc;
  ur_exp_command_buffer_update_value_arg_desc_t UpdateValDesc;
  ur_exp_command_buffer_update_kernel_launch_desc_t UpdateDesc;

  std::vector<size_t> LocalSize = {4, 4};
  const size_t SizeX = 64;
  const size_t SizeY = 64;
  std::vector<size_t> GlobalSize = {SizeX, SizeY};
  std::vector<size_t> GlobalOffset = {0, 0};
  uint32_t NDimensions = 2;

  void *Memory;
  uint32_t Val = 42;
};

struct urCommandBufferKernelHandleUpdateTest
    : uur::command_buffer::urCommandBufferMultipleKernelUpdateTest {
  virtual void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::urCommandBufferMultipleKernelUpdateTest::SetUp());

    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::checkCommandBufferUpdateSupport(
            device,
            UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_KERNEL_HANDLE));

    ur_device_usm_access_capability_flags_t shared_usm_flags;
    ASSERT_SUCCESS(
        uur::GetDeviceUSMSingleSharedSupport(device, shared_usm_flags));
    if (!(shared_usm_flags & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
      GTEST_SKIP() << "Shared USM is not supported.";
    }

    SaxpyKernel = std::make_shared<TestSaxpyKernel>(
        TestSaxpyKernel(platform, context, device));
    FillUSM2DKernel = std::make_shared<TestFill2DKernel>(
        TestFill2DKernel(platform, context, device));
    TestKernels.push_back(SaxpyKernel);
    TestKernels.push_back(FillUSM2DKernel);

    this->setUpKernels();
  }

  virtual void TearDown() override {
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::urCommandBufferMultipleKernelUpdateTest::
            TearDown());
  }

  std::shared_ptr<TestSaxpyKernel> SaxpyKernel;
  std::shared_ptr<TestFill2DKernel> FillUSM2DKernel;
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urCommandBufferKernelHandleUpdateTest);

/* Tests that it is possible to update the kernel handle of a command-buffer
 * node. This test launches a Saxpy kernel using a command-buffer and then
 * updates the node with a completely different kernel that does a fill 2D
 * operation. */
TEST_P(urCommandBufferKernelHandleUpdateTest, Success) {

  std::vector<ur_kernel_handle_t> KernelAlternatives = {
      FillUSM2DKernel->Kernel};

  ur_exp_command_buffer_command_handle_t CommandHandle;
  ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
      updatable_cmd_buf_handle, SaxpyKernel->Kernel, SaxpyKernel->NDimensions,
      &(SaxpyKernel->GlobalOffset), &(SaxpyKernel->GlobalSize),
      &(SaxpyKernel->LocalSize), KernelAlternatives.size(),
      KernelAlternatives.data(), 0, nullptr, 0, nullptr, nullptr, nullptr,
      &CommandHandle));
  ASSERT_NE(CommandHandle, nullptr);

  ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));

  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  FillUSM2DKernel->UpdateDesc.hCommand = CommandHandle;
  ASSERT_SUCCESS(urCommandBufferUpdateKernelLaunchExp(
      updatable_cmd_buf_handle, 1, &FillUSM2DKernel->UpdateDesc));
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  ASSERT_NO_FATAL_FAILURE(SaxpyKernel->validate());
  ASSERT_NO_FATAL_FAILURE(FillUSM2DKernel->validate());
}

/* Test that updates to the command kernel handle are stored in the command
 * handle */
TEST_P(urCommandBufferKernelHandleUpdateTest, UpdateAgain) {

  std::vector<ur_kernel_handle_t> KernelAlternatives = {
      FillUSM2DKernel->Kernel};

  ur_exp_command_buffer_command_handle_t CommandHandle;
  ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
      updatable_cmd_buf_handle, SaxpyKernel->Kernel, SaxpyKernel->NDimensions,
      &(SaxpyKernel->GlobalOffset), &(SaxpyKernel->GlobalSize),
      &(SaxpyKernel->LocalSize), KernelAlternatives.size(),
      KernelAlternatives.data(), 0, nullptr, 0, nullptr, nullptr, nullptr,
      &CommandHandle));
  ASSERT_NE(CommandHandle, nullptr);

  ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  FillUSM2DKernel->UpdateDesc.hCommand = CommandHandle;
  ASSERT_SUCCESS(urCommandBufferUpdateKernelLaunchExp(
      updatable_cmd_buf_handle, 1, &FillUSM2DKernel->UpdateDesc));
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  ASSERT_NO_FATAL_FAILURE(SaxpyKernel->validate());
  ASSERT_NO_FATAL_FAILURE(FillUSM2DKernel->validate());

  // If the Kernel was not stored properly in the command, then this could
  // potentially fail since it would try to use the Saxpy kernel
  FillUSM2DKernel->Val = 78;
  ASSERT_SUCCESS(urCommandBufferUpdateKernelLaunchExp(
      updatable_cmd_buf_handle, 1, &FillUSM2DKernel->UpdateDesc));
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));
  ASSERT_NO_FATAL_FAILURE(FillUSM2DKernel->validate());
}

/* Test that it is possible to change the kernel handle in a command and later
 * restore it to the original handle */
TEST_P(urCommandBufferKernelHandleUpdateTest, RestoreOriginalKernel) {

  std::vector<ur_kernel_handle_t> KernelAlternatives = {
      FillUSM2DKernel->Kernel};

  ur_exp_command_buffer_command_handle_t CommandHandle;
  ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
      updatable_cmd_buf_handle, SaxpyKernel->Kernel, SaxpyKernel->NDimensions,
      &(SaxpyKernel->GlobalOffset), &(SaxpyKernel->GlobalSize),
      &(SaxpyKernel->LocalSize), KernelAlternatives.size(),
      KernelAlternatives.data(), 0, nullptr, 0, nullptr, nullptr, nullptr,
      &CommandHandle));
  ASSERT_NE(CommandHandle, nullptr);

  ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  FillUSM2DKernel->UpdateDesc.hCommand = CommandHandle;
  ASSERT_SUCCESS(urCommandBufferUpdateKernelLaunchExp(
      updatable_cmd_buf_handle, 1, &FillUSM2DKernel->UpdateDesc));
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  ASSERT_NO_FATAL_FAILURE(SaxpyKernel->validate());
  ASSERT_NO_FATAL_FAILURE(FillUSM2DKernel->validate());

  // Updating A, so that the second launch of the saxpy kernel actually has a
  // different output.
  SaxpyKernel->A = 20;
  SaxpyKernel->UpdateDesc.hCommand = CommandHandle;
  ASSERT_SUCCESS(urCommandBufferUpdateKernelLaunchExp(
      updatable_cmd_buf_handle, 1, &SaxpyKernel->UpdateDesc));
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));
  ASSERT_NO_FATAL_FAILURE(SaxpyKernel->validate());
}

TEST_P(urCommandBufferKernelHandleUpdateTest, KernelAlternativeNotRegistered) {
  ur_exp_command_buffer_command_handle_t CommandHandle;
  ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
      updatable_cmd_buf_handle, SaxpyKernel->Kernel, SaxpyKernel->NDimensions,
      &(SaxpyKernel->GlobalOffset), &(SaxpyKernel->GlobalSize),
      &(SaxpyKernel->LocalSize), 0, nullptr, 0, nullptr, 0, nullptr, nullptr,
      nullptr, &CommandHandle));
  ASSERT_NE(CommandHandle, nullptr);

  ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));

  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));

  FillUSM2DKernel->UpdateDesc.hCommand = CommandHandle;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_VALUE,
      urCommandBufferUpdateKernelLaunchExp(updatable_cmd_buf_handle, 1,
                                           &FillUSM2DKernel->UpdateDesc));
}

TEST_P(urCommandBufferKernelHandleUpdateTest,
       RegisterInvalidKernelAlternative) {

  std::vector<ur_kernel_handle_t> KernelAlternatives = {SaxpyKernel->Kernel};

  ur_exp_command_buffer_command_handle_t CommandHandle;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_VALUE,
                   urCommandBufferAppendKernelLaunchExp(
                       updatable_cmd_buf_handle, SaxpyKernel->Kernel,
                       SaxpyKernel->NDimensions, &(SaxpyKernel->GlobalOffset),
                       &(SaxpyKernel->GlobalSize), &(SaxpyKernel->LocalSize),
                       KernelAlternatives.size(), KernelAlternatives.data(), 0,
                       nullptr, 0, nullptr, nullptr, nullptr, &CommandHandle));
}

using urCommandBufferValidUpdateParametersTest =
    urCommandBufferKernelHandleUpdateTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urCommandBufferValidUpdateParametersTest);

// Test that updating the dimensions of a kernel command does not cause an
// error.
TEST_P(urCommandBufferValidUpdateParametersTest,
       UpdateDimensionsWithoutUpdatingKernel) {

  ur_exp_command_buffer_command_handle_t CommandHandle;
  ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
      updatable_cmd_buf_handle, FillUSM2DKernel->Kernel,
      FillUSM2DKernel->NDimensions, FillUSM2DKernel->GlobalOffset.data(),
      FillUSM2DKernel->GlobalSize.data(), FillUSM2DKernel->LocalSize.data(), 0,
      nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr, &CommandHandle));
  ASSERT_NE(CommandHandle, nullptr);

  ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  ASSERT_NO_FATAL_FAILURE(FillUSM2DKernel->validate());

  size_t newGlobalWorkSize =
      FillUSM2DKernel->GlobalSize[0] * FillUSM2DKernel->GlobalSize[1];
  size_t newGlobalWorkOffset = 0;

  // Since the fill2D kernel relies on the globalID, it will still work if we
  // change the work dimensions to 1.
  FillUSM2DKernel->UpdateDesc.newWorkDim = 1;
  FillUSM2DKernel->UpdateDesc.pNewGlobalWorkSize = &newGlobalWorkSize;
  FillUSM2DKernel->UpdateDesc.pNewGlobalWorkOffset = &newGlobalWorkOffset;
  FillUSM2DKernel->UpdateDesc.hCommand = CommandHandle;
  ASSERT_SUCCESS(urCommandBufferUpdateKernelLaunchExp(
      updatable_cmd_buf_handle, 1, &FillUSM2DKernel->UpdateDesc));
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  ASSERT_NO_FATAL_FAILURE(FillUSM2DKernel->validate());
}

// Test that updating only the local work size does not cause an error.
TEST_P(urCommandBufferValidUpdateParametersTest, UpdateOnlyLocalWorkSize) {

  std::vector<ur_kernel_handle_t> KernelAlternatives = {
      FillUSM2DKernel->Kernel};

  ur_exp_command_buffer_command_handle_t CommandHandle;
  ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
      updatable_cmd_buf_handle, SaxpyKernel->Kernel, SaxpyKernel->NDimensions,
      &(SaxpyKernel->GlobalOffset), &(SaxpyKernel->GlobalSize),
      &(SaxpyKernel->LocalSize), KernelAlternatives.size(),
      KernelAlternatives.data(), 0, nullptr, 0, nullptr, nullptr, nullptr,
      &CommandHandle));
  ASSERT_NE(CommandHandle, nullptr);

  ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));

  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));

  SaxpyKernel->UpdateDesc.pNewGlobalWorkOffset = nullptr;
  SaxpyKernel->UpdateDesc.pNewGlobalWorkSize = nullptr;
  size_t newLocalSize = SaxpyKernel->LocalSize * 4;
  SaxpyKernel->UpdateDesc.pNewLocalWorkSize = &newLocalSize;
  SaxpyKernel->UpdateDesc.hCommand = CommandHandle;
  ASSERT_SUCCESS(urCommandBufferUpdateKernelLaunchExp(
      updatable_cmd_buf_handle, 1, &SaxpyKernel->UpdateDesc));
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  ASSERT_NO_FATAL_FAILURE(SaxpyKernel->validate());
}

// Tests that passing nullptr to hNewKernel works.
TEST_P(urCommandBufferValidUpdateParametersTest, SuccessNullptrHandle) {

  std::vector<ur_kernel_handle_t> KernelAlternatives = {
      FillUSM2DKernel->Kernel};

  ur_exp_command_buffer_command_handle_t CommandHandle;
  ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
      updatable_cmd_buf_handle, SaxpyKernel->Kernel, SaxpyKernel->NDimensions,
      &(SaxpyKernel->GlobalOffset), &(SaxpyKernel->GlobalSize),
      &(SaxpyKernel->LocalSize), KernelAlternatives.size(),
      KernelAlternatives.data(), 0, nullptr, 0, nullptr, nullptr, nullptr,
      &CommandHandle));
  ASSERT_NE(CommandHandle, nullptr);

  ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));

  SaxpyKernel->UpdateDesc.hNewKernel = nullptr;
  SaxpyKernel->UpdateDesc.hCommand = CommandHandle;
  ASSERT_SUCCESS(urCommandBufferUpdateKernelLaunchExp(
      updatable_cmd_buf_handle, 1, &SaxpyKernel->UpdateDesc));
  ASSERT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                           nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  ASSERT_NO_FATAL_FAILURE(SaxpyKernel->validate());
}
