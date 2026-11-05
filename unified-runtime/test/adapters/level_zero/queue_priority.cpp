// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %with-v1 ZE_ENABLE_TRACING_LAYER=1 UR_L0_EAGER_INIT=0 ./queue_priority_v1-test
// REQUIRES: v1

#include "ze_tracer_common.hpp"
#include <uur/raii.h>

#include <memory>
#include <utility>
#include <vector>

namespace {

std::vector<ze_command_queue_priority_t> queuePriorities;
std::vector<ze_command_queue_priority_t> immediateListPriorities;

void ZE_APICALL onQueueCreate(ze_command_queue_create_params_t *params,
                              ze_result_t, void *, void **) {
  queuePriorities.push_back((*params->pdesc)->priority);
}

void ZE_APICALL
onImmediateListCreate(ze_command_list_create_immediate_params_t *params,
                      ze_result_t, void *, void **) {
  immediateListPriorities.push_back((*params->paltdesc)->priority);
}

struct urQueuePriorityTest : uur::urContextTestWithParam<ur_queue_flags_t> {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam::SetUp());
    queuePriorities.clear();
    immediateListPriorities.clear();

    zel_core_callbacks_t callbacks{};
    callbacks.CommandQueue.pfnCreateCb = onQueueCreate;
    callbacks.CommandList.pfnCreateImmediateCb = onImmediateListCreate;
    tracer = enableTracing(std::move(callbacks), {});
    ASSERT_NE(tracer, nullptr);
  }

  void TearDown() override {
    if (tracer) {
      EXPECT_EQ(zelTracerSetEnabled(tracer.get(), false), ZE_RESULT_SUCCESS);
      tracer.reset();
    }
    UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam::TearDown());
  }

  void createNativeQueue(ur_queue_flags_t priorityFlags) {
    ur_queue_properties_t properties{
        UR_STRUCTURE_TYPE_QUEUE_PROPERTIES, nullptr,
        getParam() | UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE |
            priorityFlags};
    uur::raii::Queue queue;
    ASSERT_SUCCESS(urQueueCreate(context, device, &properties, queue.ptr()));

    // Force lazy native creation without submitting low-priority work that can
    // be starved by other tests: https://github.com/intel/llvm/issues/23735.
    ur_native_handle_t native = 0;
    int32_t isImmediate = -1;
    ur_queue_native_desc_t desc{UR_STRUCTURE_TYPE_QUEUE_NATIVE_DESC, nullptr,
                                &isImmediate};
    ASSERT_SUCCESS(urQueueGetNativeHandle(queue, &desc, &native));
    ASSERT_NE(native, 0);
    ASSERT_EQ(isImmediate,
              getParam() == UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE ? 1 : 0);
  }

  std::shared_ptr<_zel_tracer_handle_t> tracer;
};

UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urQueuePriorityTest,
    testing::Values(UR_QUEUE_FLAG_SUBMISSION_BATCHED,
                    UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE),
    uur::deviceTestWithParamPrinter<ur_queue_flags_t>);

TEST_P(urQueuePriorityTest, NativePrioritiesAndImmediateListRecycling) {
  for (ur_queue_flags_t priorityFlags : std::vector<ur_queue_flags_t>{
           0, 0, UR_QUEUE_FLAG_PRIORITY_LOW, UR_QUEUE_FLAG_PRIORITY_HIGH}) {
    UUR_RETURN_ON_FATAL_FAILURE(createNativeQueue(priorityFlags));
  }

  std::vector<ze_command_queue_priority_t> expected{
      ZE_COMMAND_QUEUE_PRIORITY_NORMAL, ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW,
      ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH};
  if (getParam() == UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE) {
    // The second normal-priority queue must reuse the first immediate list.
    EXPECT_EQ(immediateListPriorities, expected);
    EXPECT_TRUE(queuePriorities.empty());
  } else {
    expected.insert(expected.begin(), ZE_COMMAND_QUEUE_PRIORITY_NORMAL);
    EXPECT_EQ(queuePriorities, expected);
    EXPECT_TRUE(immediateListPriorities.empty());
  }
}

} // namespace
