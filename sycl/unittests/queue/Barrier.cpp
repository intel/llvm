//==------------------- Barrier.cpp --- queue unit tests -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

static unsigned NumOfEventsWaitWithBarrierCalls = 0;

static ur_result_t redefined_urEnqueueEventsWaitWithBarrier(void *) {
  NumOfEventsWaitWithBarrierCalls++;

  return UR_RESULT_SUCCESS;
}

TEST(Queue, HandlerBarrier) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback(
      "urEnqueueEventsWaitWithBarrier",
      &redefined_urEnqueueEventsWaitWithBarrier);
  NumOfEventsWaitWithBarrierCalls = 0;

  sycl::queue Q;

  Q.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<1>>([=]() {}); });
  Q.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<1>>([=]() {}); });

  Q.submit([&](sycl::handler &cgh) { cgh.ext_oneapi_barrier(); });

  ASSERT_EQ(NumOfEventsWaitWithBarrierCalls, 1u);
}

TEST(Queue, ExtOneAPISubmitBarrier) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback(
      "urEnqueueEventsWaitWithBarrier",
      &redefined_urEnqueueEventsWaitWithBarrier);
  NumOfEventsWaitWithBarrierCalls = 0;

  sycl::queue Q;

  Q.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<1>>([=]() {}); });
  Q.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<1>>([=]() {}); });

  Q.ext_oneapi_submit_barrier();

  ASSERT_EQ(NumOfEventsWaitWithBarrierCalls, 1u);
}

TEST(Queue, HandlerBarrierWithWaitList) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback(
      "urEnqueueEventsWaitWithBarrier",
      &redefined_urEnqueueEventsWaitWithBarrier);
  NumOfEventsWaitWithBarrierCalls = 0;

  sycl::queue Q1;
  sycl::queue Q2;
  sycl::queue Q3;

  auto E1 = Q1.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<1>>([=]() {}); });
  auto E2 = Q2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<1>>([=]() {}); });

  Q3.submit([&](sycl::handler &cgh) { cgh.ext_oneapi_barrier({E1, E2}); });

  ASSERT_EQ(NumOfEventsWaitWithBarrierCalls, 1u);
}

TEST(Queue, ExtOneAPISubmitBarrierWithWaitList) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback(
      "urEnqueueEventsWaitWithBarrier",
      &redefined_urEnqueueEventsWaitWithBarrier);
  NumOfEventsWaitWithBarrierCalls = 0;

  sycl::queue Q1;
  sycl::queue Q2;
  sycl::queue Q3;

  auto E1 = Q1.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<1>>([=]() {}); });
  auto E2 = Q2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<1>>([=]() {}); });

  Q3.ext_oneapi_submit_barrier({E1, E2});

  ASSERT_EQ(NumOfEventsWaitWithBarrierCalls, 1u);
}
