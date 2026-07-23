// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>
#include <uur/fixtures.h>
#include <uur/known_failure.h>
#include <uur/utils.h>

struct urEnqueueUSMOperationsOrderingIOQTest
    : uur::urContextTestWithParam<ur_queue_flag_t> {
  static constexpr size_t array_size = 128;

  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::urContextTestWithParam<ur_queue_flag_t>::SetUp());

    UUR_RETURN_ON_FATAL_FAILURE(uur::KernelsEnvironment::instance->LoadSource(
        "cpy_and_mult_usm", platform, il_binary));

    const ur_program_properties_t properties = {
        UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES, nullptr,
        static_cast<uint32_t>(metadatas.size()),
        metadatas.empty() ? nullptr : metadatas.data()};

    UUR_RETURN_ON_FATAL_FAILURE(
        uur::KernelsEnvironment::instance->CreateProgram(
            platform, context, device, *il_binary, &properties, &program));

    ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));

    kernel_name = uur::KernelsEnvironment::instance->GetEntryPointNames(
        "cpy_and_mult_usm")[0];
    ASSERT_FALSE(kernel_name.empty());
    ASSERT_SUCCESS(urKernelCreate(program, kernel_name.c_str(), &kernel));

    ur_queue_flags_t supported_flags = 0;
    ASSERT_SUCCESS(
        uur::GetDeviceQueueOnHostProperties(device, supported_flags));

    const ur_queue_flag_t submission_mode = getParam();
    const ur_queue_flags_t requested_flags =
        UR_QUEUE_FLAG_DISCARD_EVENTS | submission_mode;

    if ((supported_flags & UR_QUEUE_FLAG_DISCARD_EVENTS) == 0) {
      GTEST_SKIP() << "Discard events queue flag is not supported.";
    }
    if ((supported_flags & submission_mode) == 0) {
      GTEST_SKIP() << "Requested submission mode is not supported.";
    }

    ur_queue_properties_t props = {
        UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
        nullptr,
        requested_flags,
    };

    const auto result = urQueueCreate(context, device, &props, &queue);
    if (result == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
      GTEST_SKIP() << "Requested queue properties are unsupported.";
    }
    ASSERT_SUCCESS(result);
  }

  void TearDown() override {
    if (queue) {
      EXPECT_SUCCESS(urQueueRelease(queue));
    }
    if (kernel) {
      EXPECT_SUCCESS(urKernelRelease(kernel));
    }
    if (program) {
      EXPECT_SUCCESS(urProgramRelease(program));
    }
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::urContextTestWithParam<ur_queue_flag_t>::TearDown());
  }

  bool isHostUSMSupported() {
    ur_device_usm_access_capability_flags_t support = 0;
    EXPECT_SUCCESS(uur::GetDeviceUSMHostSupport(device, support));
    return (support & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS) != 0;
  }

  bool isDeviceUSMSupported() {
    ur_device_usm_access_capability_flags_t support = 0;
    EXPECT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, support));
    return (support & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS) != 0;
  }

  bool isSharedUSMSupported() {
    ur_device_usm_access_capability_flags_t single_shared_support = 0;
    ur_device_usm_access_capability_flags_t cross_shared_support = 0;
    EXPECT_SUCCESS(
        uur::GetDeviceUSMSingleSharedSupport(device, single_shared_support));
    EXPECT_SUCCESS(
        uur::GetDeviceUSMCrossSharedSupport(device, cross_shared_support));
    return ((single_shared_support | cross_shared_support) &
            UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS) != 0;
  }

  bool runForAlloc(ur_usm_type_t usm_type) {
    if (usm_type == UR_USM_TYPE_HOST && !isHostUSMSupported()) {
      return false;
    }
    if (usm_type == UR_USM_TYPE_DEVICE && !isDeviceUSMSupported()) {
      return false;
    }
    if (usm_type == UR_USM_TYPE_SHARED && !isSharedUSMSupported()) {
      return false;
    }

    void *values1 = nullptr;
    void *values2 = nullptr;
    void *values3 = nullptr;

    const size_t allocation_size = array_size * sizeof(uint32_t);

    auto alloc_one = [&](void **ptr) {
      if (usm_type == UR_USM_TYPE_HOST) {
        return urUSMHostAlloc(context, nullptr, nullptr, allocation_size, ptr);
      }
      if (usm_type == UR_USM_TYPE_DEVICE) {
        return urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                allocation_size, ptr);
      }
      return urUSMSharedAlloc(context, device, nullptr, nullptr,
                              allocation_size, ptr);
    };

    ASSERT_SUCCESS(alloc_one(&values1));
    ASSERT_SUCCESS(alloc_one(&values2));
    ASSERT_SUCCESS(alloc_one(&values3));

    std::vector<uint32_t> input(array_size);
    std::iota(input.begin(), input.end(), 0u);

    std::vector<uint32_t> tmp(array_size, 0u);
    std::vector<uint32_t> out1(array_size, 0u);
    std::vector<uint32_t> out2(array_size, 0u);
    std::vector<uint32_t> out3(array_size, 0u);

    const uint8_t zero_pattern = 0;
    const uint8_t one_pattern = 0x01;

    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, values1, input.data(),
                                      allocation_size, 0, nullptr, nullptr));
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, values2, values1,
                                      allocation_size, 0, nullptr, nullptr));
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, values3, values2,
                                      allocation_size, 0, nullptr, nullptr));

    ASSERT_SUCCESS(urEnqueueUSMFill(queue, values1, sizeof(zero_pattern),
                                    &zero_pattern, allocation_size, 0, nullptr,
                                    nullptr));

    auto ptr_arg = [](const void *ptr, uint32_t index) {
      ur_exp_kernel_arg_value_t val = {};
      val.pointer = ptr;
      ur_exp_kernel_arg_properties_t arg = {
          UR_STRUCTURE_TYPE_EXP_KERNEL_ARG_PROPERTIES,
          nullptr,
          UR_EXP_KERNEL_ARG_TYPE_POINTER,
          index,
          sizeof(void *),
          val,
      };
      return arg;
    };

    const size_t global_offset = 0;

    {
      ur_exp_kernel_arg_properties_t args[] = {ptr_arg(values3, 0),
                                               ptr_arg(values1, 1)};
      ASSERT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
          queue, kernel, 1, &global_offset, &array_size, nullptr, 2, args,
          nullptr, 0, nullptr, nullptr));
    }

    {
      ur_exp_kernel_arg_properties_t args[] = {ptr_arg(values1, 0),
                                               ptr_arg(values2, 1)};
      ASSERT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
          queue, kernel, 1, &global_offset, &array_size, nullptr, 2, args,
          nullptr, 0, nullptr, nullptr));
    }

    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, tmp.data(), values2,
                                      allocation_size, 0, nullptr, nullptr));
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, values3, tmp.data(),
                                      allocation_size, 0, nullptr, nullptr));

    {
      ur_exp_kernel_arg_properties_t args[] = {ptr_arg(values3, 0),
                                               ptr_arg(values1, 1)};
      ASSERT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
          queue, kernel, 1, &global_offset, &array_size, nullptr, 2, args,
          nullptr, 0, nullptr, nullptr));
    }

    ASSERT_SUCCESS(urEnqueueUSMFill(queue, values2, sizeof(one_pattern),
                                    &one_pattern, allocation_size, 0, nullptr,
                                    nullptr));

    {
      ur_exp_kernel_arg_properties_t args[] = {ptr_arg(values1, 0),
                                               ptr_arg(values2, 1)};
      ASSERT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
          queue, kernel, 1, &global_offset, &array_size, nullptr, 2, args,
          nullptr, 0, nullptr, nullptr));
    }

    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, out1.data(), values1,
                                      allocation_size, 0, nullptr, nullptr));
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, out2.data(), values2,
                                      allocation_size, 0, nullptr, nullptr));
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, out3.data(), values3,
                                      allocation_size, 0, nullptr, nullptr));

    ASSERT_SUCCESS(urQueueFinish(queue));

    for (size_t i = 0; i < array_size; ++i) {
      const uint32_t base = static_cast<uint32_t>(i);
      ASSERT_EQ(out1[i], base * 8u);
      ASSERT_EQ(out2[i], base * 16u);
      ASSERT_EQ(out3[i], base * 4u);
    }

    EXPECT_SUCCESS(urUSMFree(context, values1));
    EXPECT_SUCCESS(urUSMFree(context, values2));
    EXPECT_SUCCESS(urUSMFree(context, values3));
    return true;
  }

  std::shared_ptr<std::vector<char>> il_binary;
  std::vector<ur_program_metadata_t> metadatas{};
  std::string kernel_name;
  ur_program_handle_t program = nullptr;
  ur_kernel_handle_t kernel = nullptr;
  ur_queue_handle_t queue = nullptr;
};

UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urEnqueueUSMOperationsOrderingIOQTest,
    testing::Values(UR_QUEUE_FLAG_SUBMISSION_BATCHED,
                    UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE),
    uur::deviceTestWithParamPrinter<ur_queue_flag_t>);

TEST_P(urEnqueueUSMOperationsOrderingIOQTest, InOrderDiscardEventsOrdering) {
  bool any_ran = false;
  any_ran |= runForAlloc(UR_USM_TYPE_HOST);
  any_ran |= runForAlloc(UR_USM_TYPE_SHARED);
  any_ran |= runForAlloc(UR_USM_TYPE_DEVICE);

  if (!any_ran) {
    GTEST_SKIP() << "No supported USM allocation type found for this device.";
  }
}
