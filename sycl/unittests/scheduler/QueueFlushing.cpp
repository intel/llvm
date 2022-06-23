//==------------ QueueFlushing.cpp --- Scheduler unit tests ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <helpers/CommonRedefinitions.hpp>
#include <helpers/PiMock.hpp>

using namespace sycl;

static pi_queue ExpectedDepQueue = nullptr;
static bool QueueFlushed = false;
static bool EventStatusQueried = false;
static pi_event_status EventStatus = PI_EVENT_QUEUED;

static pi_result redefinedQueueFlush(pi_queue Queue) {
  EXPECT_EQ(ExpectedDepQueue, Queue);
  EXPECT_FALSE(QueueFlushed);
  QueueFlushed = true;
  EventStatus = PI_EVENT_SUBMITTED;
  return PI_SUCCESS;
}

static pi_result redefinedEventGetInfo(pi_event event, pi_event_info param_name,
                                       size_t param_value_size,
                                       void *param_value,
                                       size_t *param_value_size_ret) {
  if (param_name == PI_EVENT_INFO_COMMAND_EXECUTION_STATUS) {
    auto *Status = reinterpret_cast<pi_event_status *>(param_value);
    *Status = EventStatus;
    EventStatusQueried = true;
  }
  return PI_SUCCESS;
}

static pi_result redefinedEnqueueMemBufferReadRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_read,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  *event = reinterpret_cast<pi_event>(new int{});
  return PI_SUCCESS;
}

static pi_result redefinedEnqueueMemBufferWriteRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_write,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    const void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  *event = reinterpret_cast<pi_event>(new int{});
  return PI_SUCCESS;
}

static pi_result redefinedEnqueueMemBufferMap(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_map,
    pi_map_flags map_flags, size_t offset, size_t size,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *event, void **ret_map) {
  *event = reinterpret_cast<pi_event>(new int{});
  return PI_SUCCESS;
}

static pi_result redefinedEnqueueMemUnmap(pi_queue command_queue, pi_mem memobj,
                                          void *mapped_ptr,
                                          pi_uint32 num_events_in_wait_list,
                                          const pi_event *event_wait_list,
                                          pi_event *event) {
  *event = reinterpret_cast<pi_event>(new int{});
  return PI_SUCCESS;
}

static pi_result redefinedEnqueueMemBufferFill(
    pi_queue command_queue, pi_mem buffer, const void *pattern,
    size_t pattern_size, size_t offset, size_t size,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *event) {
  *event = reinterpret_cast<pi_event>(new int{});
  return PI_SUCCESS;
}

static void resetTestCtx() {
  EventStatus = PI_EVENT_QUEUED;
  QueueFlushed = false;
  EventStatusQueried = false;
}

static void addDepAndEnqueue(detail::Command *Cmd,
                             detail::QueueImplPtr &DepQueue,
                             detail::Requirement &MockReq) {
  MockCommand DepCmd(DepQueue);
  std::vector<detail::Command *> ToCleanUp;
  DepCmd.getEvent()->getHandleRef() = reinterpret_cast<pi_event>(new int{});
  (void)Cmd->addDep(detail::DepDesc{&DepCmd, &MockReq, nullptr}, ToCleanUp);

  detail::EnqueueResultT Res;
  MockScheduler::enqueueCommand(Cmd, Res, detail::NON_BLOCKING);
}

static void testCommandEnqueue(detail::Command *Cmd,
                               detail::QueueImplPtr &DepQueue,
                               detail::Requirement &MockReq,
                               bool ExpectedFlush = true) {
  resetTestCtx();
  addDepAndEnqueue(Cmd, DepQueue, MockReq);
  EXPECT_EQ(QueueFlushed, ExpectedFlush);
}

static void testEventStatusCheck(detail::Command *Cmd,
                                 detail::QueueImplPtr &DepQueue,
                                 detail::Requirement &MockReq,
                                 pi_event_status ReturnedEventStatus) {
  resetTestCtx();
  EventStatus = ReturnedEventStatus;
  addDepAndEnqueue(Cmd, DepQueue, MockReq);
  EXPECT_FALSE(QueueFlushed);
}

TEST_F(SchedulerTest, QueueFlushing) {
  default_selector Selector;
  platform Plt{default_selector()};
  if (Plt.is_host()) {
    std::cout << "Not run due to host-only environment\n";
    return;
  }

  unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);
  Mock.redefine<detail::PiApiKind::piQueueFlush>(redefinedQueueFlush);
  Mock.redefine<detail::PiApiKind::piEventGetInfo>(redefinedEventGetInfo);
  Mock.redefine<detail::PiApiKind::piEnqueueMemBufferReadRect>(
      redefinedEnqueueMemBufferReadRect);
  Mock.redefine<detail::PiApiKind::piEnqueueMemBufferWriteRect>(
      redefinedEnqueueMemBufferWriteRect);
  Mock.redefine<detail::PiApiKind::piEnqueueMemBufferMap>(
      redefinedEnqueueMemBufferMap);
  Mock.redefine<detail::PiApiKind::piEnqueueMemUnmap>(redefinedEnqueueMemUnmap);
  Mock.redefine<detail::PiApiKind::piEnqueueMemBufferFill>(
      redefinedEnqueueMemBufferFill);

  context Ctx{Plt};
  queue QueueA{Ctx, Selector};
  detail::QueueImplPtr QueueImplA = detail::getSyclObjImpl(QueueA);
  queue QueueB{Ctx, Selector};
  detail::QueueImplPtr QueueImplB = detail::getSyclObjImpl(QueueB);
  ExpectedDepQueue = QueueImplB->getHandleRef();

  int val;
  buffer<int, 1> Buf(&val, range<1>(1));
  detail::Requirement MockReq = getMockRequirement(Buf);
  detail::AllocaCommand AllocaCmd = detail::AllocaCommand(QueueImplA, MockReq);
  void *MockHostPtr;
  detail::EnqueueResultT Res;
  std::vector<detail::Command *> ToCleanUp;

  // Check that each of the non-blocking commands flush the dependency queue
  {
    detail::MapMemObject MapCmd{&AllocaCmd, MockReq, &MockHostPtr, QueueImplA,
                                access::mode::read_write};
    testCommandEnqueue(&MapCmd, QueueImplB, MockReq);

    detail::UnMapMemObject UnmapCmd{&AllocaCmd, MockReq, &MockHostPtr,
                                    QueueImplA};
    testCommandEnqueue(&UnmapCmd, QueueImplB, MockReq);

    device HostDevice;
    detail::QueueImplPtr DefaultHostQueue{
        new detail::queue_impl(detail::getSyclObjImpl(HostDevice), {}, {})};
    detail::AllocaCommand HostAllocaCmd =
        detail::AllocaCommand(DefaultHostQueue, MockReq);

    detail::MemCpyCommand MemCpyCmd{MockReq,    &AllocaCmd,
                                    MockReq,    &HostAllocaCmd,
                                    QueueImplA, DefaultHostQueue};
    testCommandEnqueue(&MemCpyCmd, QueueImplB, MockReq);

    detail::MemCpyCommandHost MemCpyCmdHost{MockReq,    &AllocaCmd,
                                            MockReq,    &MockHostPtr,
                                            QueueImplA, DefaultHostQueue};
    testCommandEnqueue(&MemCpyCmdHost, QueueImplB, MockReq);

    std::unique_ptr<detail::CG> CG{new detail::CGFill(/*Pattern*/ {}, &MockReq,
                                                      /*ArgsStorage*/ {},
                                                      /*AccStorage*/ {},
                                                      /*SharedPtrStorage*/ {},
                                                      /*Requirements*/ {},
                                                      /*Events*/ {})};
    detail::ExecCGCommand ExecCGCmd{std::move(CG), QueueImplA};
    MockReq.MDims = 1;
    (void)ExecCGCmd.addDep(detail::DepDesc(&AllocaCmd, &MockReq, &AllocaCmd),
                           ToCleanUp);
    testCommandEnqueue(&ExecCGCmd, QueueImplB, MockReq);
  }

  // Check dependency event without a command
  {
    resetTestCtx();
    detail::MapMemObject Cmd = {&AllocaCmd, MockReq, &MockHostPtr, QueueImplA,
                                access::mode::read_write};
    detail::EventImplPtr DepEvent{new detail::event_impl(QueueImplB)};
    DepEvent->setContextImpl(QueueImplB->getContextImplPtr());
    DepEvent->getHandleRef() = reinterpret_cast<pi_event>(new int{});
    (void)Cmd.addDep(DepEvent, ToCleanUp);
    MockScheduler::enqueueCommand(&Cmd, Res, detail::NON_BLOCKING);
    EXPECT_TRUE(QueueFlushed);
  }

  // Check that flush isn't called for a released queue.
  {
    resetTestCtx();
    detail::MapMemObject Cmd = {&AllocaCmd, MockReq, &MockHostPtr, QueueImplA,
                                access::mode::read_write};
    detail::EventImplPtr DepEvent;
    {
      queue TempQueue{Ctx, Selector};
      detail::QueueImplPtr TempQueueImpl = detail::getSyclObjImpl(TempQueue);
      DepEvent.reset(new detail::event_impl(TempQueueImpl));
      DepEvent->setContextImpl(TempQueueImpl->getContextImplPtr());
      DepEvent->getHandleRef() = reinterpret_cast<pi_event>(new int{});
    }
    (void)Cmd.addDep(DepEvent, ToCleanUp);
    MockScheduler::enqueueCommand(&Cmd, Res, detail::NON_BLOCKING);
    EXPECT_FALSE(EventStatusQueried);
    EXPECT_FALSE(QueueFlushed);
  }

  // Check that same queue dependencies are not flushed
  {
    detail::MapMemObject Cmd = {&AllocaCmd, MockReq, &MockHostPtr, QueueImplA,
                                access::mode::read_write};
    testCommandEnqueue(&Cmd, QueueImplA, MockReq, false);
  }

  // Check that flush is not called twice for the same dependency queue
  {
    resetTestCtx();
    detail::MapMemObject Cmd = {&AllocaCmd, MockReq, &MockHostPtr, QueueImplA,
                                access::mode::read_write};
    MockCommand DepCmdA(QueueImplB);
    DepCmdA.getEvent()->getHandleRef() = reinterpret_cast<pi_event>(new int{});
    (void)Cmd.addDep(detail::DepDesc{&DepCmdA, &MockReq, nullptr}, ToCleanUp);
    MockCommand DepCmdB(QueueImplB);
    DepCmdB.getEvent()->getHandleRef() = reinterpret_cast<pi_event>(new int{});
    (void)Cmd.addDep(detail::DepDesc{&DepCmdB, &MockReq, nullptr}, ToCleanUp);
    // The check is performed in redefinedQueueFlush
    MockScheduler::enqueueCommand(&Cmd, Res, detail::NON_BLOCKING);
  }

  // Check that the event status isn't requested twice for the same event
  {
    resetTestCtx();
    detail::MapMemObject CmdA{&AllocaCmd, MockReq, &MockHostPtr, QueueImplA,
                              access::mode::read_write};
    MockCommand DepCmd(QueueImplB);
    DepCmd.getEvent()->getHandleRef() = reinterpret_cast<pi_event>(new int{});
    (void)CmdA.addDep(detail::DepDesc{&DepCmd, &MockReq, nullptr}, ToCleanUp);
    MockScheduler::enqueueCommand(&CmdA, Res, detail::NON_BLOCKING);

    EventStatusQueried = false;
    detail::MapMemObject CmdB{&AllocaCmd, MockReq, &MockHostPtr, QueueImplA,
                              access::mode::read_write};
    (void)CmdB.addDep(detail::DepDesc{&DepCmd, &MockReq, nullptr}, ToCleanUp);
    MockScheduler::enqueueCommand(&CmdB, Res, detail::NON_BLOCKING);
    EXPECT_FALSE(EventStatusQueried);
  }

  // Check that flush isn't called for submitted dependencies
  {
    detail::MapMemObject CmdA{&AllocaCmd, MockReq, &MockHostPtr, QueueImplA,
                              access::mode::read_write};
    testEventStatusCheck(&CmdA, QueueImplB, MockReq, PI_EVENT_SUBMITTED);
    detail::MapMemObject CmdB{&AllocaCmd, MockReq, &MockHostPtr, QueueImplA,
                              access::mode::read_write};
    testEventStatusCheck(&CmdB, QueueImplB, MockReq, PI_EVENT_RUNNING);
    detail::MapMemObject CmdC{&AllocaCmd, MockReq, &MockHostPtr, QueueImplA,
                              access::mode::read_write};
    testEventStatusCheck(&CmdC, QueueImplB, MockReq, PI_EVENT_COMPLETE);
  }
}
