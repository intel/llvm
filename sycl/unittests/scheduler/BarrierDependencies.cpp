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

std::vector<ur_event_handle_t> HostWaitedEvents;
static ur_result_t redefinedUrEventWait(void *pParams) {
  auto params = *static_cast<ur_event_wait_params_t *>(pParams);
  for (size_t i = 0; i < *params.pnumEvents; ++i)
    HostWaitedEvents.push_back((*params.pphEventWaitList)[i]);

  return UR_RESULT_SUCCESS;
}

void clearGlobals() {
  EventsInWaitList.clear();
  BarrierEventsInWaitList.clear();
  HostWaitedEvents.clear();
  BarrierEventsWaitVisited = false;
  EventsWaitVisited = false;
}

static bool contains(const std::vector<ur_event_handle_t> &Handles,
                     ur_event_handle_t Handle) {
  return std::find(Handles.begin(), Handles.end(), Handle) != Handles.end();
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
  detail::event_impl &EventAImpl = *detail::getSyclObjImpl(EventA);
  // it means that command is enqueued
  ASSERT_NE(EventAImpl.getHandle(), nullptr);

  ASSERT_FALSE(EventsWaitVisited);
  ASSERT_TRUE(BarrierEventsWaitVisited);
  ASSERT_EQ(BarrierEventsInWaitList.size(), 0u);

  clearGlobals();
  auto EventB = QueueB.submit([&](sycl::handler &h) {
    h.depends_on(EventA);
    h.ext_oneapi_barrier();
  });
  detail::event_impl &EventBImpl = *detail::getSyclObjImpl(EventB);
  // it means that command is enqueued
  ASSERT_NE(EventBImpl.getHandle(), nullptr);

  ASSERT_TRUE(EventsWaitVisited);
  ASSERT_EQ(EventsInWaitList.size(), 1u);
  EXPECT_EQ(EventsInWaitList[0], EventAImpl.getHandle());

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
  detail::event_impl &EventAImpl = *detail::getSyclObjImpl(EventA);
  detail::event_impl &EventA2Impl = *detail::getSyclObjImpl(EventA2);
  // it means that command is enqueued
  ASSERT_NE(EventAImpl.getHandle(), nullptr);
  ASSERT_NE(EventA2Impl.getHandle(), nullptr);

  ASSERT_FALSE(EventsWaitVisited);
  ASSERT_TRUE(BarrierEventsWaitVisited);
  ASSERT_EQ(BarrierEventsInWaitList.size(), 0u);

  clearGlobals();
  auto EventB = QueueB.submit([&](sycl::handler &h) {
    h.depends_on(EventA);
    h.ext_oneapi_barrier({EventA2});
  });
  detail::event_impl &EventBImpl = *detail::getSyclObjImpl(EventB);
  // it means that command is enqueued
  ASSERT_NE(EventBImpl.getHandle(), nullptr);

  ASSERT_FALSE(EventsWaitVisited);
  ASSERT_TRUE(BarrierEventsWaitVisited);
  ASSERT_EQ(BarrierEventsInWaitList.size(), 2u);
  EXPECT_EQ(BarrierEventsInWaitList[0], EventA2Impl.getHandle());
  EXPECT_EQ(BarrierEventsInWaitList[1], EventAImpl.getHandle());

  QueueA.wait();
  QueueB.wait();
}

TEST_F(SchedulerTest, BarrierWaitListEmptyQueueShortcut) {
  sycl::unittest::UrMock<> Mock;

  sycl::platform Plt = sycl::platform();
  context Ctx{Plt};
  queue InOrderQueue{Ctx, default_selector_v, property::queue::in_order()};

  std::vector<event> DepEvents;

  sycl::event BarrierEvent = InOrderQueue.ext_oneapi_submit_barrier(DepEvents);

  InOrderQueue.wait();

  auto Info = BarrierEvent.get_info<info::event::command_execution_status>();
  ASSERT_EQ(Info, sycl::info::event_command_status::complete);
}

// An event from another context must not be passed to the backend barrier of
// the target queue. It has to be waited for on the host instead.
class BarrierCrossContextTest : public SchedulerTest {
protected:
  void SetUp() override {
    clearGlobals();
    mock::getCallbacks().set_after_callback(
        "urEnqueueEventsWaitWithBarrierExt",
        &redefinedEnqueueEventsWaitWithBarrierExt);
    mock::getCallbacks().set_after_callback("urEventWait",
                                            &redefinedUrEventWait);
  }

  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  sycl::device Dev = Plt.get_devices()[0];
  context Ctx1{Dev};
  context Ctx2{Dev};
  queue Q1{Ctx1, Dev};
  queue Q2{Ctx2, Dev};
};

TEST_F(BarrierCrossContextTest, HandlerBarrierWaitList) {
  event E1 = Q1.single_task<TestKernel>([] {});
  ur_event_handle_t E1Handle = detail::getSyclObjImpl(E1)->getHandle();
  ASSERT_NE(E1Handle, nullptr);

  event BarrierEvent =
      Q2.submit([&](handler &CGH) { CGH.ext_oneapi_barrier({E1}); });
  BarrierEvent.wait();

  EXPECT_FALSE(contains(BarrierEventsInWaitList, E1Handle));
  EXPECT_TRUE(contains(HostWaitedEvents, E1Handle));
}

TEST_F(BarrierCrossContextTest, QueueBarrierWaitList) {
  event E1 = Q1.single_task<TestKernel>([] {});
  ur_event_handle_t E1Handle = detail::getSyclObjImpl(E1)->getHandle();
  ASSERT_NE(E1Handle, nullptr);

  event BarrierEvent = Q2.ext_oneapi_submit_barrier({E1});
  BarrierEvent.wait();

  EXPECT_FALSE(contains(BarrierEventsInWaitList, E1Handle));
  EXPECT_TRUE(contains(HostWaitedEvents, E1Handle));
}

TEST_F(BarrierCrossContextTest, BarrierWaitListMixedContexts) {
  event E1 = Q1.single_task<TestKernel>([] {});
  event E2 = Q2.single_task<TestKernel>([] {});
  ur_event_handle_t E1Handle = detail::getSyclObjImpl(E1)->getHandle();
  ur_event_handle_t E2Handle = detail::getSyclObjImpl(E2)->getHandle();
  ASSERT_NE(E1Handle, nullptr);
  ASSERT_NE(E2Handle, nullptr);

  event BarrierEvent =
      Q2.submit([&](handler &CGH) { CGH.ext_oneapi_barrier({E1, E2}); });
  BarrierEvent.wait();

  ASSERT_EQ(BarrierEventsInWaitList.size(), 1u);
  EXPECT_EQ(BarrierEventsInWaitList[0], E2Handle);
  EXPECT_TRUE(contains(HostWaitedEvents, E1Handle));
}

// A NOP event from another context has nothing to wait for, so it must not
// get a connection command.
TEST_F(BarrierCrossContextTest, CrossContextNOPEventNoConnectionCmd) {
  event NopEvent = Q1.ext_oneapi_submit_barrier(std::vector<event>{});
  event E1 = Q1.single_task<TestKernel>([] {});
  detail::EventImplPtr NopImpl = detail::getSyclObjImpl(NopEvent);
  detail::EventImplPtr E1Impl = detail::getSyclObjImpl(E1);
  ASSERT_TRUE(NopImpl->isNOP());
  ASSERT_NE(E1Impl->getHandle(), nullptr);

  MockScheduler MS;
  std::vector<detail::Command *> ToEnqueue;
  auto CG = std::make_unique<detail::CGBarrier>(
      std::vector<detail::EventImplPtr>{NopImpl, E1Impl},
      ext::oneapi::experimental::event_mode_enum::none,
      detail::CG::StorageInitHelper{}, detail::CGType::BarrierWaitlist);
  detail::Command *NewCmd =
      MS.addCG(std::move(CG), &*detail::getSyclObjImpl(Q2), ToEnqueue,
               /*EventNeeded=*/true);

  // Only E1 needs a connection command.
  EXPECT_EQ(ToEnqueue.size(), 1u);

  auto &WaitList = static_cast<detail::CGBarrier &>(
                       static_cast<detail::ExecCGCommand *>(NewCmd)->getCG())
                       .MEventsWaitWithBarrier;
  ASSERT_EQ(WaitList.size(), 1u);
  EXPECT_EQ(WaitList[0], NopImpl);
}
