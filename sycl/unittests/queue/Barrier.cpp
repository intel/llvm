//==------------------- Barrier.cpp --- queue unit tests -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

static unsigned NumOfEventsWaitWithBarrierCalls = 0;
static unsigned NumEventsInWaitList = 0;

static ur_result_t redefined_urEnqueueEventsWaitWithBarrierExt(void *pParams) {
  NumOfEventsWaitWithBarrierCalls++;
  // Get the number of events in the wait list
  auto params =
      *static_cast<ur_enqueue_events_wait_with_barrier_ext_params_t *>(pParams);
  NumEventsInWaitList = *params.pnumEventsInWaitList;

  return UR_RESULT_SUCCESS;
}

TEST(Queue, HandlerBarrier) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefined_urEnqueueEventsWaitWithBarrierExt);
  NumOfEventsWaitWithBarrierCalls = 0;

  sycl::queue Q;

  Q.submit([&](sycl::handler &cgh) { cgh.single_task<TestKernel>([=]() {}); });
  Q.submit([&](sycl::handler &cgh) { cgh.single_task<TestKernel>([=]() {}); });

  Q.submit([&](sycl::handler &cgh) { cgh.ext_oneapi_barrier(); });

  ASSERT_EQ(NumOfEventsWaitWithBarrierCalls, 1u);
}

TEST(Queue, ExtOneAPISubmitBarrier) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefined_urEnqueueEventsWaitWithBarrierExt);
  NumOfEventsWaitWithBarrierCalls = 0;

  sycl::queue Q;

  Q.submit([&](sycl::handler &cgh) { cgh.single_task<TestKernel>([=]() {}); });
  Q.submit([&](sycl::handler &cgh) { cgh.single_task<TestKernel>([=]() {}); });

  Q.ext_oneapi_submit_barrier();

  ASSERT_EQ(NumOfEventsWaitWithBarrierCalls, 1u);
}

TEST(Queue, HandlerBarrierWithWaitList) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefined_urEnqueueEventsWaitWithBarrierExt);
  NumOfEventsWaitWithBarrierCalls = 0;

  sycl::queue Q1;
  sycl::queue Q2;
  sycl::queue Q3;

  auto E1 = Q1.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([=]() {}); });
  auto E2 = Q2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([=]() {}); });

  Q3.submit([&](sycl::handler &cgh) { cgh.ext_oneapi_barrier({E1, E2}); });

  ASSERT_EQ(NumOfEventsWaitWithBarrierCalls, 1u);
}

TEST(Queue, ExtOneAPISubmitBarrierWithWaitList) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefined_urEnqueueEventsWaitWithBarrierExt);
  NumOfEventsWaitWithBarrierCalls = 0;

  sycl::queue Q1;
  sycl::queue Q2;
  sycl::queue Q3;

  auto E1 = Q1.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([=]() {}); });
  auto E2 = Q2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([=]() {}); });

  Q3.ext_oneapi_submit_barrier({E1, E2});

  ASSERT_EQ(NumOfEventsWaitWithBarrierCalls, 1u);
}

TEST(Queue, BarrierWithBarrierDep) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefined_urEnqueueEventsWaitWithBarrierExt);
  sycl::queue Q1(sycl::property::queue::in_order{});
  sycl::queue Q2(sycl::property::queue::in_order{});
  Q1.submit([&](sycl::handler &cgh) { cgh.single_task<TestKernel>([=]() {}); });
  sycl::event Barrier1 = Q1.ext_oneapi_submit_barrier();
  NumEventsInWaitList = 0;
  Q2.ext_oneapi_submit_barrier({Barrier1});
  ASSERT_EQ(NumEventsInWaitList, 1u);
}
