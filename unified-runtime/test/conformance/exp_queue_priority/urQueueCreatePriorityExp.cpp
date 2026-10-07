// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/raii.h>

#include <limits>
#include <vector>

struct urQueueCreatePriorityExpTest
    : uur::urContextTestWithParam<ur_queue_flags_t> {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam::SetUp());

    ur_bool_t supported = false;
    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
        urDeviceGetInfo(device, UR_DEVICE_INFO_QUEUE_PRIORITY_SUPPORT_EXP,
                        sizeof(supported), &supported, nullptr));
    if (!supported) {
      GTEST_SKIP() << "Queue priority levels are not supported.";
    }

    ASSERT_SUCCESS(urDeviceGetInfo(device,
                                   UR_DEVICE_INFO_QUEUE_PRIORITY_HIGHEST_EXP,
                                   sizeof(highest), &highest, nullptr));
    ASSERT_SUCCESS(urDeviceGetInfo(device,
                                   UR_DEVICE_INFO_QUEUE_PRIORITY_LOWEST_EXP,
                                   sizeof(lowest), &lowest, nullptr));
  }

  ur_result_t createQueue(int32_t priority, ur_queue_flags_t extraFlags,
                          ur_queue_handle_t *phQueue) {
    ur_exp_queue_priority_properties_t priorityProps{
        UR_STRUCTURE_TYPE_EXP_QUEUE_PRIORITY_PROPERTIES, nullptr, priority};
    ur_queue_properties_t props{UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
                                &priorityProps, getParam() | extraFlags};
    return urQueueCreate(context, device, &props, phQueue);
  }

  // Checks that the queue executes work.
  void checkQueue(ur_queue_handle_t hQueue) {
    constexpr size_t size = 64;
    constexpr uint8_t pattern = 0x5A;

    void *devPtr = nullptr;
    ASSERT_SUCCESS(
        urUSMDeviceAlloc(context, device, nullptr, nullptr, size, &devPtr));
    ASSERT_SUCCESS(urEnqueueUSMFill(hQueue, devPtr, sizeof(pattern), &pattern,
                                    size, 0, nullptr, nullptr));
    std::vector<uint8_t> result(size, 0);
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(hQueue, true, result.data(), devPtr, size,
                                      0, nullptr, nullptr));
    ASSERT_SUCCESS(urUSMFree(context, devPtr));

    for (uint8_t value : result) {
      ASSERT_EQ(value, pattern);
    }
  }

  int32_t highest = 0;
  int32_t lowest = 0;
};

UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urQueueCreatePriorityExpTest,
    testing::Values(0, UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                    UR_QUEUE_FLAG_SUBMISSION_BATCHED),
    uur::deviceTestWithParamPrinter<ur_queue_flags_t>);

TEST_P(urQueueCreatePriorityExpTest, RangeContainsDefault) {
  ASSERT_LE(highest, 0);
  ASSERT_GE(lowest, 0);
}

TEST_P(urQueueCreatePriorityExpTest, SuccessAllLevels) {
  for (int32_t priority = highest; priority <= lowest; ++priority) {
    uur::raii::Queue queue;
    ASSERT_SUCCESS(createQueue(priority, 0, queue.ptr()));
    ASSERT_NE(queue, nullptr);
    UUR_RETURN_ON_FATAL_FAILURE(checkQueue(queue));
  }
}

TEST_P(urQueueCreatePriorityExpTest, SuccessOutOfRangeIsClamped) {
  for (int32_t priority : {std::numeric_limits<int32_t>::min(),
                           std::numeric_limits<int32_t>::max()}) {
    uur::raii::Queue queue;
    ASSERT_SUCCESS(createQueue(priority, 0, queue.ptr()));
    ASSERT_NE(queue, nullptr);
    UUR_RETURN_ON_FATAL_FAILURE(checkQueue(queue));
  }
}

TEST_P(urQueueCreatePriorityExpTest, SuccessWithQueueIndex) {
  ur_exp_queue_priority_properties_t priorityProps{
      UR_STRUCTURE_TYPE_EXP_QUEUE_PRIORITY_PROPERTIES, nullptr, highest};
  ur_queue_index_properties_t indexProps{
      UR_STRUCTURE_TYPE_QUEUE_INDEX_PROPERTIES, &priorityProps, 0};
  ur_queue_properties_t props{UR_STRUCTURE_TYPE_QUEUE_PROPERTIES, &indexProps,
                              getParam()};

  uur::raii::Queue queue;
  ASSERT_SUCCESS(urQueueCreate(context, device, &props, queue.ptr()));
  ASSERT_NE(queue, nullptr);
  UUR_RETURN_ON_FATAL_FAILURE(checkQueue(queue));
}

TEST_P(urQueueCreatePriorityExpTest, InvalidWithPriorityFlags) {
  for (ur_queue_flags_t flag :
       {UR_QUEUE_FLAG_PRIORITY_LOW, UR_QUEUE_FLAG_PRIORITY_HIGH}) {
    ur_queue_handle_t queue = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES,
                     createQueue(0, flag, &queue));
    ASSERT_EQ(queue, nullptr);
  }
}
