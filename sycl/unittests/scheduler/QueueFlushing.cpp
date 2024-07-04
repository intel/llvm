//==------------ QueueFlushing.cpp --- Scheduler unit tests ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

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

static pi_result redefinedEventGetInfoAfter(pi_event event,
                                            pi_event_info param_name,
                                            size_t param_value_size,
                                            void *param_value,
                                            size_t *param_value_size_ret) {
  EXPECT_NE(event, nullptr);
  if (param_name == PI_EVENT_INFO_COMMAND_EXECUTION_STATUS) {
    auto *Status = reinterpret_cast<pi_event_status *>(param_value);
    *Status = EventStatus;
    EventStatusQueried = true;
  }
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

  pi_event PIEvent = nullptr;
  pi_result CallRet = mock_piEventCreate(/*pi_context=*/0x0, &PIEvent);
  EXPECT_TRUE(PI_SUCCESS == CallRet);

  DepCmd.getEvent()->getHandleRef() = PIEvent;
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
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  Mock.redefineBefore<detail::PiApiKind::piQueueFlush>(redefinedQueueFlush);
  Mock.redefineAfter<detail::PiApiKind::piEventGetInfo>(
      redefinedEventGetInfoAfter);

  context Ctx{Plt};
  queue QueueA{Ctx, default_selector_v};
  detail::QueueImplPtr QueueImplA = detail::getSyclObjImpl(QueueA);
  queue QueueB{Ctx, default_selector_v};
  detail::QueueImplPtr QueueImplB = detail::getSyclObjImpl(QueueB);
  ExpectedDepQueue = QueueImplB->getHandleRef();

  int val;
  buffer<int, 1> Buf(&val, range<1>(1));
  detail::Requirement MockReq = getMockRequirement(Buf);

  pi_mem PIBuf = nullptr;
  pi_result Ret = mock_piMemBufferCreate(/*pi_context=*/0x0,
                                         PI_MEM_FLAGS_ACCESS_RW, /*size=*/1,
                                         /*host_ptr=*/nullptr, &PIBuf);
  EXPECT_TRUE(Ret == PI_SUCCESS);

  detail::AllocaCommand AllocaCmd = detail::AllocaCommand(QueueImplA, MockReq);
  AllocaCmd.MMemAllocation = PIBuf;
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

    detail::AllocaCommand HostAllocaCmd =
        detail::AllocaCommand(nullptr, MockReq);

    detail::MemCpyCommand MemCpyCmd{MockReq,        &AllocaCmd, MockReq,
                                    &HostAllocaCmd, QueueImplA, nullptr};
    testCommandEnqueue(&MemCpyCmd, QueueImplB, MockReq);

    detail::MemCpyCommandHost MemCpyCmdHost{MockReq,      &AllocaCmd, MockReq,
                                            &MockHostPtr, QueueImplA, nullptr};
    testCommandEnqueue(&MemCpyCmdHost, QueueImplB, MockReq);

    std::unique_ptr<detail::CG> CG{
        new detail::CGFill(/*Pattern*/ {}, &MockReq,
                           detail::CG::StorageInitHelper(
                               /*ArgsStorage*/ {},
                               /*AccStorage*/ {},
                               /*SharedPtrStorage*/ {},
                               /*Requirements*/ {},
                               /*Events*/ {}))};
    detail::ExecCGCommand ExecCGCmd{std::move(CG), QueueImplA,
                                    /*EventNeeded=*/true};
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

    pi_event PIEvent = nullptr;
    pi_result CallRet = mock_piEventCreate(/*pi_context=*/0x0, &PIEvent);
    EXPECT_TRUE(PI_SUCCESS == CallRet);

    DepEvent->getHandleRef() = PIEvent;
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
      queue TempQueue{Ctx, default_selector_v};
      detail::QueueImplPtr TempQueueImpl = detail::getSyclObjImpl(TempQueue);
      DepEvent.reset(new detail::event_impl(TempQueueImpl));
      DepEvent->setContextImpl(TempQueueImpl->getContextImplPtr());

      pi_event PIEvent = nullptr;
      pi_result CallRet = mock_piEventCreate(/*pi_context=*/0x0, &PIEvent);
      EXPECT_TRUE(PI_SUCCESS == CallRet);

      DepEvent->getHandleRef() = PIEvent;
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

    pi_event PIEvent = nullptr;
    pi_result CallRet = mock_piEventCreate(/*pi_context=*/0x0, &PIEvent);
    EXPECT_TRUE(PI_SUCCESS == CallRet);

    DepCmdA.getEvent()->getHandleRef() = PIEvent;
    (void)Cmd.addDep(detail::DepDesc{&DepCmdA, &MockReq, nullptr}, ToCleanUp);
    MockCommand DepCmdB(QueueImplB);

    PIEvent = nullptr;
    CallRet = mock_piEventCreate(/*pi_context=*/0x0, &PIEvent);
    EXPECT_TRUE(PI_SUCCESS == CallRet);

    DepCmdB.getEvent()->getHandleRef() = PIEvent;
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

    pi_event PIEvent = nullptr;
    pi_result CallRet = mock_piEventCreate(/*pi_context=*/0x0, &PIEvent);
    EXPECT_TRUE(PI_SUCCESS == CallRet);

    DepCmd.getEvent()->getHandleRef() = PIEvent;
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

  // Check that nullptr pi_events are handled correctly.
  {
    resetTestCtx();
    detail::MapMemObject CmdA{&AllocaCmd, MockReq, &MockHostPtr, QueueImplA,
                              access::mode::read_write};
    MockCommand DepCmd(QueueImplB);
    (void)CmdA.addDep(detail::DepDesc{&DepCmd, &MockReq, nullptr}, ToCleanUp);
    MockScheduler::enqueueCommand(&CmdA, Res, detail::NON_BLOCKING);
    EXPECT_FALSE(EventStatusQueried);
  }
}
