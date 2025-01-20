// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/known_failure.h>

struct urEnqueueUSMAdviseWithParamTest
    : uur::urUSMDeviceAllocTestWithParam<ur_usm_advice_flag_t> {
  void SetUp() override {
    // The setup for the parent fixture does a urQueueFlush, which isn't
    // supported by native cpu.
    UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

    UUR_RETURN_ON_FATAL_FAILURE(
        uur::urUSMDeviceAllocTestWithParam<ur_usm_advice_flag_t>::SetUp());
  }
};
UUR_DEVICE_TEST_SUITE_P(urEnqueueUSMAdviseWithParamTest,
                        ::testing::Values(UR_USM_ADVICE_FLAG_DEFAULT),
                        uur::deviceTestWithParamPrinter<ur_usm_advice_flag_t>);

TEST_P(urEnqueueUSMAdviseWithParamTest, Success) {
  // HIP and CUDA return UR_RESULT_ERROR_ADAPTER_SPECIFIC to issue a warning
  // about the hint being unsupported.
  // TODO: codify this in the spec and account for it in the CTS.
  UUR_KNOWN_FAILURE_ON(uur::HIP{}, uur::CUDA{});

  ur_event_handle_t advise_event = nullptr;
  ASSERT_SUCCESS(urEnqueueUSMAdvise(queue, ptr, allocation_size, getParam(),
                                    &advise_event));

  ASSERT_NE(advise_event, nullptr);
  ASSERT_SUCCESS(urQueueFlush(queue));
  ASSERT_SUCCESS(urEventWait(1, &advise_event));

  ur_event_status_t advise_event_status{};
  ASSERT_SUCCESS(
      urEventGetInfo(advise_event, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS,
                     sizeof(ur_event_status_t), &advise_event_status, nullptr));
  EXPECT_EQ(advise_event_status, UR_EVENT_STATUS_COMPLETE);
  ASSERT_SUCCESS(urEventRelease(advise_event));
}

struct urEnqueueUSMAdviseTest : uur::urUSMDeviceAllocTest {
  void SetUp() override {
    UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});
    uur::urUSMDeviceAllocTest::SetUp();
  }
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueUSMAdviseTest);

TEST_P(urEnqueueUSMAdviseTest, MultipleParamsSuccess) {
  // HIP and CUDA return UR_RESULT_ERROR_ADAPTER_SPECIFIC to issue a warning
  // about the hint being unsupported.
  // TODO: codify this in the spec and account for it in the CTS.
  UUR_KNOWN_FAILURE_ON(uur::HIP{}, uur::CUDA{});

  ASSERT_SUCCESS(urEnqueueUSMAdvise(queue, ptr, allocation_size,
                                    UR_USM_ADVICE_FLAG_SET_READ_MOSTLY |
                                        UR_USM_ADVICE_FLAG_BIAS_CACHED,
                                    nullptr));
}

TEST_P(urEnqueueUSMAdviseTest, InvalidNullHandleQueue) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urEnqueueUSMAdvise(nullptr, ptr, allocation_size,
                                      UR_USM_ADVICE_FLAG_DEFAULT, nullptr));
}

TEST_P(urEnqueueUSMAdviseTest, InvalidNullPointerMem) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urEnqueueUSMAdvise(queue, nullptr, allocation_size,
                                      UR_USM_ADVICE_FLAG_DEFAULT, nullptr));
}

TEST_P(urEnqueueUSMAdviseTest, InvalidEnumeration) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                   urEnqueueUSMAdvise(queue, ptr, allocation_size,
                                      UR_USM_ADVICE_FLAG_FORCE_UINT32,
                                      nullptr));
}

TEST_P(urEnqueueUSMAdviseTest, InvalidSizeZero) {
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_SIZE,
      urEnqueueUSMAdvise(queue, ptr, 0, UR_USM_ADVICE_FLAG_DEFAULT, nullptr));
}

TEST_P(urEnqueueUSMAdviseTest, InvalidSizeTooLarge) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                   urEnqueueUSMAdvise(queue, ptr, allocation_size * 2,
                                      UR_USM_ADVICE_FLAG_DEFAULT, nullptr));
}

TEST_P(urEnqueueUSMAdviseTest, NonCoherentDeviceMemorySuccessOrWarning) {
  ur_result_t result =
      urEnqueueUSMAdvise(queue, ptr, allocation_size,
                         UR_USM_ADVICE_FLAG_SET_NON_COHERENT_MEMORY, nullptr);
  ASSERT_EQ(result,
            result & (UR_RESULT_SUCCESS | UR_RESULT_ERROR_ADAPTER_SPECIFIC));
}
