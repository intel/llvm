//==-------- BarrierDependencies.cpp --- Scheduler unit tests --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>

#include <detail/event_impl.hpp>

#include <gtest/gtest.h>

#include <sycl/sycl.hpp>

using namespace sycl;

std::vector<ur_event_handle_t> EventsInWaitList;
bool EventsWaitVisited = false;
static ur_result_t redefinedEventWait(void *pParams) {
  EventsWaitVisited = true;

  auto params = *static_cast<ur_enqueue_events_wait_params_t *>(pParams);
  for (size_t i = 0; i < *params.pnumEventsInWaitList; ++i)
    EventsInWaitList.push_back((*params.pphEventWaitList)[i]);

  return UR_RESULT_SUCCESS;
}

std::vector<ur_event_handle_t> BarrierEventsInWaitList;
bool BarrierEventsWaitVisited = false;
ur_result_t redefinedEnqueueEventsWaitWithBarrierExt(void *pParams) {
  BarrierEventsWaitVisited = true;

  auto params =
      *static_cast<ur_enqueue_events_wait_with_barrier_ext_params_t *>(pParams);
  for (auto i = 0u; i < *params.pnumEventsInWaitList; i++) {
    BarrierEventsInWaitList.push_back((*params.pphEventWaitList)[i]);
  }
  return UR_RESULT_SUCCESS;
}

void clearGlobals() {
  EventsInWaitList.clear();
  BarrierEventsInWaitList.clear();
  BarrierEventsWaitVisited = false;
  EventsWaitVisited = false;
}

TEST_F(SchedulerTest, BarrierWithDependsOn) {
  clearGlobals();

  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  mock::getCallbacks().set_after_callback("urEnqueueEventsWait",
                                          &redefinedEventWait);
  mock::getCallbacks().set_after_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefinedEnqueueEventsWaitWithBarrierExt);

  context Ctx{Plt};
  queue QueueA{Ctx, default_selector_v, property::queue::in_order()};
  queue QueueB{Ctx, default_selector_v, property::queue::in_order()};

  auto EventA =
      QueueA.submit([&](sycl::handler &h) { h.ext_oneapi_barrier(); });
  std::shared_ptr<detail::event_impl> EventAImpl =
      detail::getSyclObjImpl(EventA);
  // it means that command is enqueued
  ASSERT_NE(EventAImpl->getHandle(), nullptr);

  ASSERT_FALSE(EventsWaitVisited);
  ASSERT_TRUE(BarrierEventsWaitVisited);
  ASSERT_EQ(BarrierEventsInWaitList.size(), 0u);

  clearGlobals();
  auto EventB = QueueB.submit([&](sycl::handler &h) {
    h.depends_on(EventA);
    h.ext_oneapi_barrier();
  });
  std::shared_ptr<detail::event_impl> EventBImpl =
      detail::getSyclObjImpl(EventB);
  // it means that command is enqueued
  ASSERT_NE(EventBImpl->getHandle(), nullptr);

  ASSERT_TRUE(EventsWaitVisited);
  ASSERT_EQ(EventsInWaitList.size(), 1u);
  EXPECT_EQ(EventsInWaitList[0], EventAImpl->getHandle());

  ASSERT_TRUE(BarrierEventsWaitVisited);
  ASSERT_EQ(BarrierEventsInWaitList.size(), 0u);

  QueueA.wait();
  QueueB.wait();
}

TEST_F(SchedulerTest, BarrierWaitListWithDependsOn) {
  clearGlobals();

  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  mock::getCallbacks().set_after_callback("urEnqueueEventsWait",
                                          &redefinedEventWait);
  mock::getCallbacks().set_after_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefinedEnqueueEventsWaitWithBarrierExt);

  context Ctx{Plt};
  queue QueueA{Ctx, default_selector_v, property::queue::in_order()};
  queue QueueB{Ctx, default_selector_v, property::queue::in_order()};

  auto EventA =
      QueueA.submit([&](sycl::handler &h) { h.ext_oneapi_barrier(); });
  auto EventA2 =
      QueueA.submit([&](sycl::handler &h) { h.ext_oneapi_barrier(); });
  std::shared_ptr<detail::event_impl> EventAImpl =
      detail::getSyclObjImpl(EventA);
  std::shared_ptr<detail::event_impl> EventA2Impl =
      detail::getSyclObjImpl(EventA2);
  // it means that command is enqueued
  ASSERT_NE(EventAImpl->getHandle(), nullptr);
  ASSERT_NE(EventA2Impl->getHandle(), nullptr);

  ASSERT_FALSE(EventsWaitVisited);
  ASSERT_TRUE(BarrierEventsWaitVisited);
  ASSERT_EQ(BarrierEventsInWaitList.size(), 0u);

  clearGlobals();
  auto EventB = QueueB.submit([&](sycl::handler &h) {
    h.depends_on(EventA);
    h.ext_oneapi_barrier({EventA2});
  });
  std::shared_ptr<detail::event_impl> EventBImpl =
      detail::getSyclObjImpl(EventB);
  // it means that command is enqueued
  ASSERT_NE(EventBImpl->getHandle(), nullptr);

  ASSERT_FALSE(EventsWaitVisited);
  ASSERT_TRUE(BarrierEventsWaitVisited);
  ASSERT_EQ(BarrierEventsInWaitList.size(), 2u);
  EXPECT_EQ(BarrierEventsInWaitList[0], EventA2Impl->getHandle());
  EXPECT_EQ(BarrierEventsInWaitList[1], EventAImpl->getHandle());

  QueueA.wait();
  QueueB.wait();
}
