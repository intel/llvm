//==------------------- Barrier.cpp --- queue unit tests -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <helpers/CommandSubmitWrappers.hpp>
#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

static unsigned NumOfEventsWaitWithBarrierCalls = 0;
static unsigned NumEventsInWaitList = 0;

class Queue : public testing::TestWithParam<bool> {};

static ur_result_t redefined_urEnqueueEventsWaitWithBarrierExt(void *pParams) {
  NumOfEventsWaitWithBarrierCalls++;
  // Get the number of events in the wait list
  auto params =
      *static_cast<ur_enqueue_events_wait_with_barrier_ext_params_t *>(pParams);
  NumEventsInWaitList = *params.pnumEventsInWaitList;

  return UR_RESULT_SUCCESS;
}

TEST_P(Queue, HandlerBarrier) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefined_urEnqueueEventsWaitWithBarrierExt);
  NumOfEventsWaitWithBarrierCalls = 0;
  bool UseShortcutFunction = GetParam();

  sycl::queue Q;

  sycl::unittest::single_task_wrapper<TestKernel>(UseShortcutFunction, Q,
                                                  [=]() {});
  sycl::unittest::single_task_wrapper<TestKernel>(UseShortcutFunction, Q,
                                                  [=]() {});

  Q.submit([&](sycl::handler &cgh) { cgh.ext_oneapi_barrier(); });

  ASSERT_EQ(NumOfEventsWaitWithBarrierCalls, 1u);
}

TEST_P(Queue, ExtOneAPISubmitBarrier) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefined_urEnqueueEventsWaitWithBarrierExt);
  NumOfEventsWaitWithBarrierCalls = 0;
  bool UseShortcutFunction = GetParam();

  sycl::queue Q;

  sycl::unittest::single_task_wrapper<TestKernel>(UseShortcutFunction, Q,
                                                  [=]() {});
  sycl::unittest::single_task_wrapper<TestKernel>(UseShortcutFunction, Q,
                                                  [=]() {});

  Q.ext_oneapi_submit_barrier();

  ASSERT_EQ(NumOfEventsWaitWithBarrierCalls, 1u);
}

TEST_P(Queue, HandlerBarrierWithWaitList) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefined_urEnqueueEventsWaitWithBarrierExt);
  NumOfEventsWaitWithBarrierCalls = 0;
  bool UseShortcutFunction = GetParam();

  sycl::queue Q1;
  sycl::queue Q2;
  sycl::queue Q3;

  auto E1 = sycl::unittest::single_task_wrapper<TestKernel>(UseShortcutFunction,
                                                            Q1, [=]() {});
  auto E2 = sycl::unittest::single_task_wrapper<TestKernel>(UseShortcutFunction,
                                                            Q2, [=]() {});

  Q3.submit([&](sycl::handler &cgh) { cgh.ext_oneapi_barrier({E1, E2}); });

  ASSERT_EQ(NumOfEventsWaitWithBarrierCalls, 1u);
}

TEST_P(Queue, ExtOneAPISubmitBarrierWithWaitList) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefined_urEnqueueEventsWaitWithBarrierExt);
  NumOfEventsWaitWithBarrierCalls = 0;
  bool UseShortcutFunction = GetParam();

  sycl::queue Q1;
  sycl::queue Q2;
  sycl::queue Q3;

  auto E1 = sycl::unittest::single_task_wrapper<TestKernel>(UseShortcutFunction,
                                                            Q1, [=]() {});
  auto E2 = sycl::unittest::single_task_wrapper<TestKernel>(UseShortcutFunction,
                                                            Q2, [=]() {});

  Q3.ext_oneapi_submit_barrier({E1, E2});

  ASSERT_EQ(NumOfEventsWaitWithBarrierCalls, 1u);
}

TEST_P(Queue, BarrierWithBarrierDep) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefined_urEnqueueEventsWaitWithBarrierExt);
  bool UseShortcutFunction = GetParam();

  sycl::queue Q1(sycl::property::queue::in_order{});
  sycl::queue Q2(sycl::property::queue::in_order{});

  sycl::unittest::single_task_wrapper<TestKernel>(UseShortcutFunction, Q1,
                                                  [=]() {});

  sycl::event Barrier1 = Q1.ext_oneapi_submit_barrier();
  NumEventsInWaitList = 0;
  Q2.ext_oneapi_submit_barrier({Barrier1});
  ASSERT_EQ(NumEventsInWaitList, 1u);
}

INSTANTIATE_TEST_SUITE_P(
    QueueTestInstance, Queue,
    testing::Values(
        true,
        false)); /* Whether to use the shortcut command submission function */
