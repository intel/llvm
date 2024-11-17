//==------------ QueueFlushing.cpp --- Scheduler unit tests ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"
#include "detail/event_impl.hpp"
#include "ur_mock_helpers.hpp"

#include <helpers/UrMock.hpp>

using namespace sycl;

static ur_queue_handle_t ExpectedDepQueue = nullptr;
static bool QueueFlushed = false;
static bool EventStatusQueried = false;
static ur_event_status_t EventStatus = UR_EVENT_STATUS_QUEUED;

static ur_result_t redefinedQueueFlush(void *pParams) {
  auto params = *static_cast<ur_queue_flush_params_t *>(pParams);
  EXPECT_EQ(ExpectedDepQueue, *params.phQueue);
  EXPECT_FALSE(QueueFlushed);
  QueueFlushed = true;
  EventStatus = UR_EVENT_STATUS_SUBMITTED;
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedEventGetInfoAfter(void *pParams) {
  auto params = *static_cast<ur_event_get_info_params_t *>(pParams);
  EXPECT_NE(*params.phEvent, nullptr);
  if (*params.ppropName == UR_EVENT_INFO_COMMAND_EXECUTION_STATUS) {
    auto *Status = reinterpret_cast<ur_event_status_t *>(*params.ppPropValue);
    *Status = EventStatus;
    EventStatusQueried = true;
  }
  return UR_RESULT_SUCCESS;
}

static void resetTestCtx() {
  EventStatus = UR_EVENT_STATUS_QUEUED;
  QueueFlushed = false;
  EventStatusQueried = false;
}

static void addDepAndEnqueue(detail::Command *Cmd,
                             detail::QueueImplPtr &DepQueue,
                             detail::Requirement &MockReq) {
  MockCommand DepCmd(DepQueue);
  std::vector<detail::Command *> ToCleanUp;

  ur_event_handle_t UREvent = mock::createDummyHandle<ur_event_handle_t>();

  DepCmd.getEvent()->setHandle(UREvent);
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
                                 ur_event_status_t ReturnedEventStatus) {
  resetTestCtx();
  EventStatus = ReturnedEventStatus;
  addDepAndEnqueue(Cmd, DepQueue, MockReq);
  EXPECT_FALSE(QueueFlushed);
}

TEST_F(SchedulerTest, QueueFlushing) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  mock::getCallbacks().set_before_callback("urQueueFlush",
                                           &redefinedQueueFlush);
  mock::getCallbacks().set_after_callback("urEventGetInfo",
                                          &redefinedEventGetInfoAfter);

  context Ctx{Plt};
  queue QueueA{Ctx, default_selector_v};
  detail::QueueImplPtr QueueImplA = detail::getSyclObjImpl(QueueA);
  queue QueueB{Ctx, default_selector_v};
  detail::QueueImplPtr QueueImplB = detail::getSyclObjImpl(QueueB);
  ExpectedDepQueue = QueueImplB->getHandleRef();

  int val;
  buffer<int, 1> Buf(&val, range<1>(1));
  detail::Requirement MockReq = getMockRequirement(Buf);

  ur_mem_handle_t URBuf = mock::createDummyHandle<ur_mem_handle_t>();

  detail::AllocaCommand AllocaCmd = detail::AllocaCommand(QueueImplA, MockReq);
  AllocaCmd.MMemAllocation = URBuf;
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

    ur_event_handle_t UREvent = mock::createDummyHandle<ur_event_handle_t>();

    DepEvent->setHandle(UREvent);
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

      ur_event_handle_t UREvent = mock::createDummyHandle<ur_event_handle_t>();

      DepEvent->setHandle(UREvent);
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

    ur_event_handle_t UREvent = mock::createDummyHandle<ur_event_handle_t>();

    DepCmdA.getEvent()->setHandle(UREvent);
    (void)Cmd.addDep(detail::DepDesc{&DepCmdA, &MockReq, nullptr}, ToCleanUp);
    MockCommand DepCmdB(QueueImplB);

    UREvent = mock::createDummyHandle<ur_event_handle_t>();

    DepCmdB.getEvent()->setHandle(UREvent);
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

    ur_event_handle_t UREvent = mock::createDummyHandle<ur_event_handle_t>();

    DepCmd.getEvent()->setHandle(UREvent);
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
    testEventStatusCheck(&CmdA, QueueImplB, MockReq, UR_EVENT_STATUS_SUBMITTED);
    detail::MapMemObject CmdB{&AllocaCmd, MockReq, &MockHostPtr, QueueImplA,
                              access::mode::read_write};
    testEventStatusCheck(&CmdB, QueueImplB, MockReq, UR_EVENT_STATUS_RUNNING);
    detail::MapMemObject CmdC{&AllocaCmd, MockReq, &MockHostPtr, QueueImplA,
                              access::mode::read_write};
    testEventStatusCheck(&CmdC, QueueImplB, MockReq, UR_EVENT_STATUS_COMPLETE);
  }

  // Check that nullptr UR event handles are handled correctly.
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
