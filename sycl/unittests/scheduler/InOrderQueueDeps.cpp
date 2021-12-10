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

#include <iostream>
#include <memory>

using namespace cl::sycl;

static pi_result
redefinedMemBufferCreate(pi_context context, pi_mem_flags flags, size_t size,
                         void *host_ptr, pi_mem *ret_mem,
                         const pi_mem_properties *properties = nullptr) {
  return PI_SUCCESS;
}

static pi_result redefinedMemRelease(pi_mem mem) { return PI_SUCCESS; }

static pi_result redefinedEnqueueMemBufferReadRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_read,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  EXPECT_EQ(num_events_in_wait_list, 0u);
  *event = reinterpret_cast<pi_event>(1);
  return PI_SUCCESS;
}

static pi_result redefinedEnqueueMemBufferWriteRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_write,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    const void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  EXPECT_EQ(num_events_in_wait_list, 0u);
  *event = reinterpret_cast<pi_event>(1);
  return PI_SUCCESS;
}

static pi_result redefinedEnqueueMemBufferMap(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_map,
    pi_map_flags map_flags, size_t offset, size_t size,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *event, void **ret_map) {
  EXPECT_EQ(num_events_in_wait_list, 0u);
  *event = reinterpret_cast<pi_event>(1);
  return PI_SUCCESS;
}

static pi_result redefinedEnqueueMemUnmap(pi_queue command_queue, pi_mem memobj,
                                          void *mapped_ptr,
                                          pi_uint32 num_events_in_wait_list,
                                          const pi_event *event_wait_list,
                                          pi_event *event) {
  EXPECT_EQ(num_events_in_wait_list, 0u);
  *event = reinterpret_cast<pi_event>(1);
  return PI_SUCCESS;
}

static pi_result redefinedEventsWait(pi_uint32 num_events,
                                     const pi_event *event_list) {
  return PI_SUCCESS;
}

pi_result redefinedEventRelease(pi_event event) { return PI_SUCCESS; }

TEST_F(SchedulerTest, InOrderQueueDeps) {
  default_selector Selector;
  platform Plt{default_selector()};
  if (Plt.is_host()) {
    std::cout << "Not run due to host-only environment\n";
    return;
  }

  unittest::PiMock Mock{Plt};
  Mock.redefine<detail::PiApiKind::piMemBufferCreate>(redefinedMemBufferCreate);
  Mock.redefine<detail::PiApiKind::piMemRelease>(redefinedMemRelease);
  Mock.redefine<detail::PiApiKind::piEnqueueMemBufferReadRect>(
      redefinedEnqueueMemBufferReadRect);
  Mock.redefine<detail::PiApiKind::piEnqueueMemBufferWriteRect>(
      redefinedEnqueueMemBufferWriteRect);
  Mock.redefine<detail::PiApiKind::piEnqueueMemBufferMap>(
      redefinedEnqueueMemBufferMap);
  Mock.redefine<detail::PiApiKind::piEnqueueMemUnmap>(redefinedEnqueueMemUnmap);
  Mock.redefine<detail::PiApiKind::piEventsWait>(redefinedEventsWait);
  Mock.redefine<detail::PiApiKind::piEventRelease>(redefinedEventRelease);

  context Ctx{Plt.get_devices()[0]};
  queue InOrderQueue{Ctx, Selector, property::queue::in_order()};
  cl::sycl::detail::QueueImplPtr InOrderQueueImpl =
      detail::getSyclObjImpl(InOrderQueue);

  device HostDevice;
  std::shared_ptr<detail::queue_impl> DefaultHostQueue{
      new detail::queue_impl(detail::getSyclObjImpl(HostDevice), {}, {})};

  MockScheduler MS;

  int val;
  buffer<int, 1> Buf(&val, range<1>(1));
  detail::Requirement Req = getMockRequirement(Buf);

  std::vector<detail::Command *> AuxCmds;
  detail::MemObjRecord *Record =
      MS.getOrInsertMemObjRecord(InOrderQueueImpl, &Req, AuxCmds);
  MS.getOrCreateAllocaForReq(Record, &Req, InOrderQueueImpl, AuxCmds);
  MS.getOrCreateAllocaForReq(Record, &Req, DefaultHostQueue, AuxCmds);

  // Check that sequential memory movements submitted to the same in-order
  // queue do not depend on each other.
  detail::Command *Cmd =
      MS.insertMemoryMove(Record, &Req, DefaultHostQueue, AuxCmds);
  detail::EnqueueResultT Res;
  auto ReadLock = MS.acquireGraphReadLock();
  MockScheduler::enqueueCommand(Cmd, Res, detail::NON_BLOCKING);
  Cmd = MS.insertMemoryMove(Record, &Req, InOrderQueueImpl, AuxCmds);
  MockScheduler::enqueueCommand(Cmd, Res, detail::NON_BLOCKING);
  Cmd = MS.insertMemoryMove(Record, &Req, DefaultHostQueue, AuxCmds);
  MockScheduler::enqueueCommand(Cmd, Res, detail::NON_BLOCKING);
}
