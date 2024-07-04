//==------------ InOrderQueueueueDeps.cpp --- Scheduler unit tests ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <helpers/PiMock.hpp>
#include <helpers/TestKernel.hpp>

#include <iostream>
#include <memory>

namespace {
using namespace sycl;

pi_result redefinedEnqueueMemBufferReadRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_read,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  EXPECT_EQ(num_events_in_wait_list, 0u);
  return PI_SUCCESS;
}

pi_result redefinedEnqueueMemBufferWriteRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_write,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    const void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  EXPECT_EQ(num_events_in_wait_list, 0u);
  return PI_SUCCESS;
}

pi_result redefinedEnqueueMemBufferMap(pi_queue command_queue, pi_mem buffer,
                                       pi_bool blocking_map,
                                       pi_map_flags map_flags, size_t offset,
                                       size_t size,
                                       pi_uint32 num_events_in_wait_list,
                                       const pi_event *event_wait_list,
                                       pi_event *event, void **ret_map) {
  EXPECT_EQ(num_events_in_wait_list, 0u);
  return PI_SUCCESS;
}

pi_result redefinedEnqueueMemUnmap(pi_queue command_queue, pi_mem memobj,
                                   void *mapped_ptr,
                                   pi_uint32 num_events_in_wait_list,
                                   const pi_event *event_wait_list,
                                   pi_event *event) {
  EXPECT_EQ(num_events_in_wait_list, 0u);
  return PI_SUCCESS;
}

TEST_F(SchedulerTest, InOrderQueueDeps) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  Mock.redefineBefore<detail::PiApiKind::piEnqueueMemBufferReadRect>(
      redefinedEnqueueMemBufferReadRect);
  Mock.redefineBefore<detail::PiApiKind::piEnqueueMemBufferWriteRect>(
      redefinedEnqueueMemBufferWriteRect);
  Mock.redefineBefore<detail::PiApiKind::piEnqueueMemBufferMap>(
      redefinedEnqueueMemBufferMap);
  Mock.redefineBefore<detail::PiApiKind::piEnqueueMemUnmap>(
      redefinedEnqueueMemUnmap);

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
pi_event ExpectedEvent = nullptr;
pi_result redefinedEnqueueEventsWaitWithBarrier(
    pi_queue command_queue, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  EXPECT_EQ(num_events_in_wait_list, 1u);
  EXPECT_EQ(ExpectedEvent, *event_wait_list);
  BarrierCalled = true;
  return PI_SUCCESS;
}

sycl::event submitKernel(sycl::queue &Q) {
  return Q.submit(
      [&](handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
}

TEST_F(SchedulerTest, InOrderQueueIsolatedDeps) {
  // Check that isolated kernels (i.e. those that don't modify the graph)
  // are handled properly during filtering.
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  Mock.redefineBefore<detail::PiApiKind::piEnqueueEventsWaitWithBarrier>(
      redefinedEnqueueEventsWaitWithBarrier);

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
    ExpectedEvent = detail::getSyclObjImpl(E2)->getHandleRef();
    Q1.ext_oneapi_submit_barrier({E1, E2});
    EXPECT_TRUE(BarrierCalled);
  }
}
} // anonymous namespace
