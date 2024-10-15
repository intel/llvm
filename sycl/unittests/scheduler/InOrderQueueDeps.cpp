//==------------ InOrderQueueueueDeps.cpp --- Scheduler unit tests ---------==//
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
#include <sycl/usm.hpp>

#include <iostream>
#include <memory>

namespace {
using namespace sycl;

ur_result_t redefinedEnqueueMemBufferReadRect(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_mem_buffer_read_rect_params_t *>(pParams);
  EXPECT_EQ(*params.pnumEventsInWaitList, 0u);
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedEnqueueMemBufferWriteRect(void *pParams) {
  auto params = *static_cast<ur_enqueue_mem_buffer_write_params_t *>(pParams);
  EXPECT_EQ(*params.pnumEventsInWaitList, 0u);
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedEnqueueMemBufferMap(void *pParams) {
  auto params = *static_cast<ur_enqueue_mem_buffer_map_params_t *>(pParams);
  EXPECT_EQ(*params.pnumEventsInWaitList, 0u);
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedEnqueueMemUnmap(void *pParams) {
  auto params = *static_cast<ur_enqueue_mem_unmap_params_t *>(pParams);
  EXPECT_EQ(*params.pnumEventsInWaitList, 0u);
  return UR_RESULT_SUCCESS;
}

TEST_F(SchedulerTest, InOrderQueueDeps) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  mock::getCallbacks().set_before_callback("urEnqueueMemBufferReadRect",
                                           &redefinedEnqueueMemBufferReadRect);
  mock::getCallbacks().set_before_callback("urEnqueueMemBufferWriteRect",
                                           &redefinedEnqueueMemBufferWriteRect);
  mock::getCallbacks().set_before_callback("urEnqueueMemBufferMap",
                                           &redefinedEnqueueMemBufferMap);
  mock::getCallbacks().set_before_callback("urEnqueueMemUnmap",
                                           &redefinedEnqueueMemUnmap);

  context Ctx{Plt.get_devices()[0]};
  queue InOrderQueue{Ctx, default_selector_v, property::queue::in_order()};
  sycl::detail::QueueImplPtr InOrderQueueImpl =
      detail::getSyclObjImpl(InOrderQueue);

  MockScheduler MS;

  int val;
  buffer<int, 1> Buf(&val, range<1>(1));
  detail::Requirement Req = getMockRequirement(Buf);

  detail::MemObjRecord *Record =
      MS.getOrInsertMemObjRecord(InOrderQueueImpl, &Req);
  std::vector<detail::Command *> AuxCmds;
  MS.getOrCreateAllocaForReq(Record, &Req, InOrderQueueImpl, AuxCmds);
  MS.getOrCreateAllocaForReq(Record, &Req, nullptr, AuxCmds);

  // Check that sequential memory movements submitted to the same in-order
  // queue do not depend on each other.
  detail::Command *Cmd = MS.insertMemoryMove(Record, &Req, nullptr, AuxCmds);
  detail::EnqueueResultT Res;
  auto ReadLock = MS.acquireGraphReadLock();
  MockScheduler::enqueueCommand(Cmd, Res, detail::NON_BLOCKING);
  Cmd = MS.insertMemoryMove(Record, &Req, InOrderQueueImpl, AuxCmds);
  MockScheduler::enqueueCommand(Cmd, Res, detail::NON_BLOCKING);
  Cmd = MS.insertMemoryMove(Record, &Req, nullptr, AuxCmds);
  MockScheduler::enqueueCommand(Cmd, Res, detail::NON_BLOCKING);
}

bool BarrierCalled = false;
ur_event_handle_t ExpectedEvent = nullptr;
ur_result_t redefinedEnqueueEventsWaitWithBarrier(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_events_wait_with_barrier_params_t *>(pParams);
  EXPECT_EQ(*params.pnumEventsInWaitList, 1u);
  EXPECT_EQ(ExpectedEvent, **params.pphEventWaitList);
  BarrierCalled = true;
  return UR_RESULT_SUCCESS;
}

sycl::event submitKernel(sycl::queue &Q) {
  return Q.submit(
      [&](handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
}

TEST_F(SchedulerTest, InOrderQueueIsolatedDeps) {
  // Check that isolated kernels (i.e. those that don't modify the graph)
  // are handled properly during filtering.
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  mock::getCallbacks().set_before_callback(
      "urEnqueueEventsWaitWithBarrier", &redefinedEnqueueEventsWaitWithBarrier);
  BarrierCalled = false;

  context Ctx{Plt.get_devices()[0]};
  queue Q1{Ctx, default_selector_v, property::queue::in_order()};
  {
    event E = submitKernel(Q1);
    Q1.ext_oneapi_submit_barrier({E});
    EXPECT_FALSE(BarrierCalled);
  }
  queue Q2{Ctx, default_selector_v, property::queue::in_order()};
  {
    event E1 = submitKernel(Q1);
    event E2 = submitKernel(Q2);
    ExpectedEvent = detail::getSyclObjImpl(E2)->getHandle();
    Q1.ext_oneapi_submit_barrier({E1, E2});
    EXPECT_TRUE(BarrierCalled);
  }
}

std::vector<size_t> KernelEventListSize;

inline ur_result_t customEnqueueKernelLaunch(void *pParams) {
  auto params = *static_cast<ur_enqueue_kernel_launch_params_t *>(pParams);
  KernelEventListSize.push_back(*params.pnumEventsInWaitList);
  return UR_RESULT_SUCCESS;
}

TEST_F(SchedulerTest, TwoInOrderQueuesOnSameContext) {
  KernelEventListSize.clear();
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback("urEnqueueKernelLaunch",
                                           &customEnqueueKernelLaunch);

  sycl::platform Plt = sycl::platform();

  context Ctx{Plt};
  queue InOrderQueueFirst{Ctx, default_selector_v, property::queue::in_order()};
  queue InOrderQueueSecond{Ctx, default_selector_v,
                           property::queue::in_order()};

  event EvFirst = InOrderQueueFirst.submit(
      [&](sycl::handler &CGH) { CGH.single_task<TestKernel<>>([] {}); });
  std::ignore = InOrderQueueSecond.submit([&](sycl::handler &CGH) {
    CGH.depends_on(EvFirst);
    CGH.single_task<TestKernel<>>([] {});
  });

  InOrderQueueFirst.wait();
  InOrderQueueSecond.wait();

  ASSERT_EQ(KernelEventListSize.size(), 2u);
  EXPECT_EQ(KernelEventListSize[0] /*EventsCount*/, 0u);
  EXPECT_EQ(KernelEventListSize[1] /*EventsCount*/, 1u);
}

TEST_F(SchedulerTest, InOrderQueueNoSchedulerPath) {
  KernelEventListSize.clear();
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback("urEnqueueKernelLaunch",
                                           &customEnqueueKernelLaunch);

  sycl::platform Plt = sycl::platform();

  context Ctx{Plt};
  queue InOrderQueue{Ctx, default_selector_v, property::queue::in_order()};

  event EvFirst = InOrderQueue.submit(
      [&](sycl::handler &CGH) { CGH.single_task<TestKernel<>>([] {}); });
  std::ignore = InOrderQueue.submit([&](sycl::handler &CGH) {
    CGH.depends_on(EvFirst);
    CGH.single_task<TestKernel<>>([] {});
  });

  InOrderQueue.wait();

  ASSERT_EQ(KernelEventListSize.size(), 2u);
  EXPECT_EQ(KernelEventListSize[0] /*EventsCount*/, 0u);
  // native device events for device kernel submitted to the same in-order queue
  // don't need to be explicitly passed as dependencies
  EXPECT_EQ(KernelEventListSize[1] /*EventsCount*/, 0u);
}

// Test that barrier is not filtered out when waitlist contains an event
// produced by command which is bypassing the scheduler.
TEST_F(SchedulerTest, BypassSchedulerWithBarrier) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  mock::getCallbacks().set_before_callback(
      "urEnqueueEventsWaitWithBarrier", &redefinedEnqueueEventsWaitWithBarrier);
  BarrierCalled = false;

  context Ctx{Plt};
  queue Q1{Ctx, default_selector_v, property::queue::in_order()};
  queue Q2{Ctx, default_selector_v, property::queue::in_order()};
  static constexpr size_t Size = 10;

  int *X = malloc_host<int>(Size, Ctx);

  // Submit a command which bypasses the scheduler.
  auto FillEvent = Q2.memset(X, 0, sizeof(int) * Size);
  // Submit a barrier which depends on that event.
  ExpectedEvent = detail::getSyclObjImpl(FillEvent)->getHandle();
  auto BarrierQ1 = Q1.ext_oneapi_submit_barrier({FillEvent});
  Q1.wait();
  Q2.wait();
  // Verify that barrier is not filtered out.
  EXPECT_EQ(BarrierCalled, true);

  free(X, Ctx);
}

} // anonymous namespace
