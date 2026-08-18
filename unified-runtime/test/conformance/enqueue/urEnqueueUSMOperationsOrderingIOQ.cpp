// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urEnqueueUSMOperationsOrderingIOQTest
    : uur::urContextTestWithParam<ur_queue_flag_t> {
  static constexpr size_t array_size = 128;

  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::urContextTestWithParam<ur_queue_flag_t>::SetUp());

    UUR_RETURN_ON_FATAL_FAILURE(uur::KernelsEnvironment::instance->LoadSource(
        "discard_events_ordering_usm", platform, il_binary));

    const ur_program_properties_t properties = {
        UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES, nullptr,
        static_cast<uint32_t>(metadatas.size()),
        metadatas.empty() ? nullptr : metadatas.data()};

    UUR_RETURN_ON_FATAL_FAILURE(
        uur::KernelsEnvironment::instance->CreateProgram(
            platform, context, device, *il_binary, &properties, &program));

    ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));

    auto entry_points = uur::KernelsEnvironment::instance->GetEntryPointNames(
        "discard_events_ordering_usm");
    ASSERT_FALSE(entry_points.empty());
    kernel_name = entry_points[0];
    ASSERT_FALSE(kernel_name.empty());
    ASSERT_SUCCESS(urKernelCreate(program, kernel_name.c_str(), &kernel));

    const ur_queue_flag_t submission_mode = getParam();
    const ur_queue_flags_t requested_flags =
        UR_QUEUE_FLAG_DISCARD_EVENTS | submission_mode;

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
      ASSERT_SUCCESS(urQueueRelease(queue));
    }
    if (kernel) {
      ASSERT_SUCCESS(urKernelRelease(kernel));
    }
    if (program) {
      ASSERT_SUCCESS(urProgramRelease(program));
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

    const auto res1 = alloc_one(&values1);
    const auto res2 = alloc_one(&values2);
    const auto res3 = alloc_one(&values3);
    EXPECT_SUCCESS(res1);
    EXPECT_SUCCESS(res2);
    EXPECT_SUCCESS(res3);
    if (res1 != UR_RESULT_SUCCESS || res2 != UR_RESULT_SUCCESS ||
        res3 != UR_RESULT_SUCCESS || !values1 || !values2 || !values3) {
      if (values1) {
        (void)urUSMFree(context, values1);
      }
      if (values2) {
        (void)urUSMFree(context, values2);
      }
      if (values3) {
        (void)urUSMFree(context, values3);
      }
      return true;
    }

    std::vector<uint32_t> input(array_size);
    std::iota(input.begin(), input.end(), 0u);

    std::vector<uint32_t> tmp(array_size, 0u);
    std::vector<uint32_t> out1(array_size, 0u);
    std::vector<uint32_t> out2(array_size, 0u);
    std::vector<uint32_t> out3(array_size, 0u);

    const uint8_t zero_pattern = 0;

    EXPECT_SUCCESS(urEnqueueUSMMemcpy(queue, false, values1, input.data(),
                                      allocation_size, 0, nullptr, nullptr));
    EXPECT_SUCCESS(urEnqueueUSMMemcpy(queue, false, values2, values1,
                                      allocation_size, 0, nullptr, nullptr));
    EXPECT_SUCCESS(urEnqueueUSMMemcpy(queue, false, values3, values2,
                                      allocation_size, 0, nullptr, nullptr));

    EXPECT_SUCCESS(urEnqueueUSMFill(queue, values1, sizeof(zero_pattern),
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

    const size_t global_offset[] = {0};
    const size_t global_size[] = {array_size};

    {
      ur_exp_kernel_arg_properties_t args[] = {
          ptr_arg(values1, 0), ptr_arg(values2, 1), ptr_arg(values3, 2)};
      EXPECT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
          queue, kernel, 1, global_offset, global_size, nullptr, 3, args,
          nullptr, 0, nullptr, nullptr));
    }

    {
      ur_exp_kernel_arg_properties_t args[] = {
          ptr_arg(values1, 0), ptr_arg(values2, 1), ptr_arg(values3, 2)};
      EXPECT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
          queue, kernel, 1, global_offset, global_size, nullptr, 3, args,
          nullptr, 0, nullptr, nullptr));
    }

    EXPECT_SUCCESS(urEnqueueUSMMemcpy(queue, false, tmp.data(), values1,
                                      allocation_size, 0, nullptr, nullptr));
    EXPECT_SUCCESS(urEnqueueUSMMemcpy(queue, false, values2, tmp.data(),
                                      allocation_size, 0, nullptr, nullptr));

    {
      ur_exp_kernel_arg_properties_t args[] = {
          ptr_arg(values1, 0), ptr_arg(values2, 1), ptr_arg(values3, 2)};
      EXPECT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
          queue, kernel, 1, global_offset, global_size, nullptr, 3, args,
          nullptr, 0, nullptr, nullptr));
    }

    {
      ur_exp_kernel_arg_properties_t args[] = {
          ptr_arg(values1, 0), ptr_arg(values2, 1), ptr_arg(values3, 2)};
      EXPECT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
          queue, kernel, 1, global_offset, global_size, nullptr, 3, args,
          nullptr, 0, nullptr, nullptr));
    }

    {
      ur_exp_kernel_arg_properties_t args[] = {
          ptr_arg(values1, 0), ptr_arg(values2, 1), ptr_arg(values3, 2)};
      EXPECT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
          queue, kernel, 1, global_offset, global_size, nullptr, 3, args,
          nullptr, 0, nullptr, nullptr));
    }

    EXPECT_SUCCESS(urEnqueueUSMMemcpy(queue, false, out1.data(), values1,
                                      allocation_size, 0, nullptr, nullptr));
    EXPECT_SUCCESS(urEnqueueUSMMemcpy(queue, false, out2.data(), values2,
                                      allocation_size, 0, nullptr, nullptr));
    EXPECT_SUCCESS(urEnqueueUSMMemcpy(queue, false, out3.data(), values3,
                                      allocation_size, 0, nullptr, nullptr));

    EXPECT_SUCCESS(urQueueFinish(queue));

    for (size_t i = 0; i < array_size; ++i) {
      const uint32_t base = static_cast<uint32_t>(i);
      EXPECT_EQ(out1[i], base + 11110u);
      EXPECT_EQ(out2[i], base);
      EXPECT_EQ(out3[i], base);
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
