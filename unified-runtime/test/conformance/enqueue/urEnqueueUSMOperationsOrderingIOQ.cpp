// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../device_code/discard_events_ordering_usm_consts.h"
#include <uur/fixtures.h>

// Test parameter combining submission mode and batch size.
// Tests in-order execution with discard_events optimization across different
// batching configurations to ensure batching doesn't break ordering semantics.
struct QueueParameter {
  ur_queue_flag_t submission_mode;
  uint32_t batch_size;

  QueueParameter(ur_queue_flag_t mode, uint32_t size)
      : submission_mode(mode), batch_size(size) {}
};

inline std::string PrintQueueParam(
    const testing::TestParamInfo<std::tuple<uur::DeviceTuple, QueueParameter>>
        &info) {
  const auto &device = std::get<0>(info.param).device;
  const QueueParameter &queue_param = std::get<1>(info.param);
  std::string mode_str =
      (queue_param.submission_mode == UR_QUEUE_FLAG_SUBMISSION_BATCHED)
          ? "Batched"
          : "Immediate";
  return uur::GetPlatformAndDeviceName(device) + "__" +
         uur::GTestSanitizeString(mode_str + "_BatchSize_" +
                                  std::to_string(queue_param.batch_size));
}

struct urEnqueueUSMOperationsOrderingIOQTest
    : uur::urContextTestWithParam<QueueParameter> {
  static constexpr size_t array_size = 128;

  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::urContextTestWithParam<QueueParameter>::SetUp());

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

    const QueueParameter params = getParam();
    const ur_queue_flag_t submission_mode = params.submission_mode;
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
        uur::urContextTestWithParam<QueueParameter>::TearDown());
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

  bool isUSMTypeSupported(ur_usm_type_t usm_type) {
    switch (usm_type) {
    case UR_USM_TYPE_HOST:
      return isHostUSMSupported();
    case UR_USM_TYPE_DEVICE:
      return isDeviceUSMSupported();
    case UR_USM_TYPE_SHARED:
      return isSharedUSMSupported();
    default:
      return false;
    }
  }

  // Execute the in-order execution test for a specific USM allocation type.
  // Returns true if test ran successfully, false if allocation or execution failed.
  bool runOrderingTestForUSMType(ur_usm_type_t usm_type) {

    EXPECT_TRUE(isUSMTypeSupported(usm_type))
        << "Attempting to run test with unsupported USM type";

    auto usm_deleter = [this](void *ptr) {
      if (ptr) {
        (void)urUSMFree(context, ptr);
      }
    };

    void *values1_raw = nullptr;
    void *values2_raw = nullptr;
    void *values3_raw = nullptr;

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

    const auto res1 = alloc_one(&values1_raw);
    const auto res2 = alloc_one(&values2_raw);
    const auto res3 = alloc_one(&values3_raw);
    EXPECT_SUCCESS(res1);
    EXPECT_SUCCESS(res2);
    EXPECT_SUCCESS(res3);

    std::unique_ptr<void, decltype(usm_deleter)> values1(values1_raw,
                                                         usm_deleter);
    std::unique_ptr<void, decltype(usm_deleter)> values2(values2_raw,
                                                         usm_deleter);
    std::unique_ptr<void, decltype(usm_deleter)> values3(values3_raw,
                                                         usm_deleter);

    if (res1 != UR_RESULT_SUCCESS || res2 != UR_RESULT_SUCCESS ||
        res3 != UR_RESULT_SUCCESS || !values1 || !values2 || !values3) {
      return false;
    }

    std::vector<uint32_t> input(array_size);
    std::iota(input.begin(), input.end(), 0u);

    std::vector<uint32_t> tmp(array_size, 0u);
    std::vector<uint32_t> out1(array_size, 0u);
    std::vector<uint32_t> out2(array_size, 0u);
    std::vector<uint32_t> out3(array_size, 0u);

    const uint8_t zero_pattern = 0;

    EXPECT_SUCCESS(urEnqueueUSMMemcpy(queue, false, values1.get(), input.data(),
                                      allocation_size, 0, nullptr, nullptr));
    EXPECT_SUCCESS(urEnqueueUSMMemcpy(queue, false, values2.get(),
                                      values1.get(), allocation_size, 0,
                                      nullptr, nullptr));
    EXPECT_SUCCESS(urEnqueueUSMMemcpy(queue, false, values3.get(),
                                      values2.get(), allocation_size, 0,
                                      nullptr, nullptr));

    EXPECT_SUCCESS(urEnqueueUSMFill(queue, values1.get(), sizeof(zero_pattern),
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

    auto enqueue_kernel_with_pointers = [&](void *p1, void *p2, void *p3) {
      ur_exp_kernel_arg_properties_t args[] = {ptr_arg(p1, 0), ptr_arg(p2, 1),
                                               ptr_arg(p3, 2)};
      EXPECT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
          queue, kernel, 1, global_offset, global_size, nullptr, 3, args,
          nullptr, 0, nullptr, nullptr));
    };

    enqueue_kernel_with_pointers(values1.get(), values2.get(), values3.get());
    enqueue_kernel_with_pointers(values1.get(), values2.get(), values3.get());

    EXPECT_SUCCESS(urEnqueueUSMMemcpy(queue, false, tmp.data(), values1.get(),
                                      allocation_size, 0, nullptr, nullptr));
    EXPECT_SUCCESS(urEnqueueUSMMemcpy(queue, false, values2.get(), tmp.data(),
                                      allocation_size, 0, nullptr, nullptr));

    enqueue_kernel_with_pointers(values1.get(), values2.get(), values3.get());
    enqueue_kernel_with_pointers(values1.get(), values2.get(), values3.get());
    enqueue_kernel_with_pointers(values1.get(), values2.get(), values3.get());

    EXPECT_SUCCESS(urEnqueueUSMMemcpy(queue, false, out1.data(), values1.get(),
                                      allocation_size, 0, nullptr, nullptr));
    EXPECT_SUCCESS(urEnqueueUSMMemcpy(queue, false, out2.data(), values2.get(),
                                      allocation_size, 0, nullptr, nullptr));
    EXPECT_SUCCESS(urEnqueueUSMMemcpy(queue, false, out3.data(), values3.get(),
                                      allocation_size, 0, nullptr, nullptr));

    EXPECT_SUCCESS(urQueueFinish(queue));

    static constexpr uint32_t DISCARD_EVENTS_EXPECTED_FINAL_INCREMENT =
        DISCARD_EVENTS_STAGE_2_INCREMENT + DISCARD_EVENTS_STAGE_3_INCREMENT +
        DISCARD_EVENTS_STAGE_4_INCREMENT + DISCARD_EVENTS_STAGE_5_INCREMENT;

    for (size_t i = 0; i < array_size; ++i) {
      const uint32_t base = static_cast<uint32_t>(i);
      // Verify all ordering stages executed successfully.
      // out1 reaches this value only if all 5 stages completed in order.
      EXPECT_EQ(out1[i], base + DISCARD_EVENTS_EXPECTED_FINAL_INCREMENT);
      EXPECT_EQ(out2[i], base);
      EXPECT_EQ(out3[i], base);
    }

    return true;
  }

  // Legacy wrapper for compatibility.
  bool runForAlloc(ur_usm_type_t usm_type) {
    if (!isUSMTypeSupported(usm_type)) {
      return false;
    }
    return runOrderingTestForUSMType(usm_type);
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
    testing::Values(QueueParameter(UR_QUEUE_FLAG_SUBMISSION_BATCHED, 0),
                    QueueParameter(UR_QUEUE_FLAG_SUBMISSION_BATCHED, 1),
                    QueueParameter(UR_QUEUE_FLAG_SUBMISSION_BATCHED, 2),
                    QueueParameter(UR_QUEUE_FLAG_SUBMISSION_BATCHED, 3),
                    QueueParameter(UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE, 0),
                    QueueParameter(UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE, 1),
                    QueueParameter(UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE, 2),
                    QueueParameter(UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE, 3)),
    PrintQueueParam);

TEST_P(urEnqueueUSMOperationsOrderingIOQTest, InOrderDiscardEventsOrdering) {
  bool any_ran = false;
  any_ran |= runForAlloc(UR_USM_TYPE_HOST);
  any_ran |= runForAlloc(UR_USM_TYPE_SHARED);
  any_ran |= runForAlloc(UR_USM_TYPE_DEVICE);

  if (!any_ran) {
    GTEST_SKIP() << "No supported USM allocation type found for this device.";
  }
}
