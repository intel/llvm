//==------------ InOrderQueueueueDeps.cpp --- Scheduler unit tests ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <helpers/CommandSubmitWrappers.hpp>
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

TEST_P(SchedulerTest, InOrderQueueDeps) {
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
  detail::queue_impl &InOrderQueueImpl = *detail::getSyclObjImpl(InOrderQueue);

  MockScheduler MS;

  int val;
  buffer<int, 1> Buf(&val, range<1>(1));
  detail::Requirement Req = getMockRequirement(Buf);

  detail::MemObjRecord *Record =
      MS.getOrInsertMemObjRecord(&InOrderQueueImpl, &Req);
  std::vector<detail::Command *> AuxCmds;
  MS.getOrCreateAllocaForReq(Record, &Req, &InOrderQueueImpl, AuxCmds);
  MS.getOrCreateAllocaForReq(Record, &Req, nullptr, AuxCmds);

  // Check that sequential memory movements submitted to the same in-order
  // queue do not depend on each other.
  detail::Command *Cmd = MS.insertMemoryMove(Record, &Req, nullptr, AuxCmds);
  detail::EnqueueResultT Res;
  auto ReadLock = MS.acquireGraphReadLock();
  MockScheduler::enqueueCommand(Cmd, Res, detail::NON_BLOCKING);
  Cmd = MS.insertMemoryMove(Record, &Req, &InOrderQueueImpl, AuxCmds);
  MockScheduler::enqueueCommand(Cmd, Res, detail::NON_BLOCKING);
  Cmd = MS.insertMemoryMove(Record, &Req, nullptr, AuxCmds);
  MockScheduler::enqueueCommand(Cmd, Res, detail::NON_BLOCKING);
}

bool BarrierCalled = false;
ur_event_handle_t ExpectedEvent = nullptr;
ur_result_t redefinedEnqueueEventsWaitWithBarrierExt(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_events_wait_with_barrier_ext_params_t *>(pParams);
  EXPECT_EQ(*params.pnumEventsInWaitList, 1u);
  EXPECT_EQ(ExpectedEvent, **params.pphEventWaitList);
  BarrierCalled = true;
  return UR_RESULT_SUCCESS;
}

TEST_P(SchedulerTest, InOrderQueueIsolatedDeps) {
  // Check that isolated kernels (i.e. those that don't modify the graph)
  // are handled properly during filtering.
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  bool UseShortcutFunction = GetParam();
  mock::getCallbacks().set_before_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefinedEnqueueEventsWaitWithBarrierExt);
  BarrierCalled = false;

  context Ctx{Plt.get_devices()[0]};
  queue Q1{Ctx, default_selector_v, property::queue::in_order()};
  {
    event E = sycl::unittest::single_task_wrapper<TestKernel>(
        UseShortcutFunction, Q1, []() {});
    Q1.ext_oneapi_submit_barrier({E});
    EXPECT_FALSE(BarrierCalled);
  }
  queue Q2{Ctx, default_selector_v, property::queue::in_order()};
  {
    event E1 = sycl::unittest::single_task_wrapper<TestKernel>(
        UseShortcutFunction, Q1, []() {});
    event E2 = sycl::unittest::single_task_wrapper<TestKernel>(
        UseShortcutFunction, Q2, []() {});
    ExpectedEvent = detail::getSyclObjImpl(E2)->getHandle();
    Q1.ext_oneapi_submit_barrier({E1, E2});
    EXPECT_TRUE(BarrierCalled);
  }
}

std::vector<size_t> KernelEventListSize;

inline ur_result_t customEnqueueKernelLaunchWithArgsExp(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_kernel_launch_with_args_exp_params_t *>(pParams);
  KernelEventListSize.push_back(*params.pnumEventsInWaitList);
  return UR_RESULT_SUCCESS;
}

TEST_P(SchedulerTest, TwoInOrderQueuesOnSameContext) {
  KernelEventListSize.clear();
  sycl::unittest::UrMock<> Mock;
  bool UseShortcutFunction = GetParam();
  mock::getCallbacks().set_before_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &customEnqueueKernelLaunchWithArgsExp);

  sycl::platform Plt = sycl::platform();

  context Ctx{Plt};
  queue InOrderQueueFirst{Ctx, default_selector_v, property::queue::in_order()};
  queue InOrderQueueSecond{Ctx, default_selector_v,
                           property::queue::in_order()};

  event EvFirst = sycl::unittest::single_task_wrapper<TestKernel>(
      UseShortcutFunction, InOrderQueueFirst, []() {});
  std::ignore = sycl::unittest::single_task_wrapper<TestKernel>(
      UseShortcutFunction, InOrderQueueSecond, EvFirst, []() {});

  InOrderQueueFirst.wait();
  InOrderQueueSecond.wait();

  ASSERT_EQ(KernelEventListSize.size(), 2u);
  EXPECT_EQ(KernelEventListSize[0] /*EventsCount*/, 0u);
  EXPECT_EQ(KernelEventListSize[1] /*EventsCount*/, 1u);
}

TEST_P(SchedulerTest, InOrderQueueNoSchedulerPath) {
  KernelEventListSize.clear();
  sycl::unittest::UrMock<> Mock;
  bool UseShortcutFunction = GetParam();
  mock::getCallbacks().set_before_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &customEnqueueKernelLaunchWithArgsExp);

  sycl::platform Plt = sycl::platform();

  context Ctx{Plt};
  queue InOrderQueue{Ctx, default_selector_v, property::queue::in_order()};

  event EvFirst = sycl::unittest::single_task_wrapper<TestKernel>(
      UseShortcutFunction, InOrderQueue, []() {});
  std::ignore = sycl::unittest::single_task_wrapper<TestKernel>(
      UseShortcutFunction, InOrderQueue, EvFirst, []() {});

  InOrderQueue.wait();

  ASSERT_EQ(KernelEventListSize.size(), 2u);
  EXPECT_EQ(KernelEventListSize[0] /*EventsCount*/, 0u);
  // native device events for device kernel submitted to the same in-order queue
  // don't need to be explicitly passed as dependencies
  EXPECT_EQ(KernelEventListSize[1] /*EventsCount*/, 0u);
}

} // anonymous namespace

INSTANTIATE_TEST_SUITE_P(
    SchedulerTestInstance, SchedulerTest,
    testing::Values(
        true,
        false)); /* Whether to use the shortcut command submission function */
