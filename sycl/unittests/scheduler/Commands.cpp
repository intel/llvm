//==----------- Commands.cpp --- Commands unit tests -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"
#include <detail/buffer_impl.hpp>
#include <helpers/PiMock.hpp>

#include <iostream>

using namespace sycl;

pi_result redefinePiEnqueueEventsWaitWithBarrier(pi_queue Queue,
                                                 pi_uint32 NumEventsInWaitList,
                                                 const pi_event *EventWaitList,
                                                 pi_event *Event) {

  for (pi_uint32 i = 0; i != NumEventsInWaitList; ++i)
    EXPECT_NE(EventWaitList[i], nullptr);

  return PI_SUCCESS;
}

// Hack that allows to return a context in redefinePiEventGetInfo
sycl::detail::pi::PiContext queue_global_context = nullptr;

pi_result redefinePiEventGetInfo(pi_event, pi_event_info, size_t,
                                 void *param_value, size_t *) {
  *reinterpret_cast<sycl::detail::pi::PiContext *>(param_value) =
      queue_global_context;
  return PI_SUCCESS;
}

//
// This test checks a handling of empty events in WaitWithBarrier command.
// Original reproducer for l0 plugin led to segfault(nullptr dereference):
//
// #include <sycl/sycl.hpp>
// int main() {
//     sycl::queue q;
//     sycl::event e;
//     q.submit_barrier({e});
// }
//
TEST_F(SchedulerTest, WaitEmptyEventWithBarrier) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();

  Mock.redefineBefore<detail::PiApiKind::piEnqueueEventsWaitWithBarrier>(
      redefinePiEnqueueEventsWaitWithBarrier);

  queue Queue{Plt.get_devices()[0]};
  sycl::detail::QueueImplPtr QueueImpl = detail::getSyclObjImpl(Queue);

  queue_global_context =
      detail::getSyclObjImpl(Queue.get_context())->getHandleRef();

  Mock.redefineBefore<detail::PiApiKind::piEventGetInfo>(
      redefinePiEventGetInfo);

  auto EmptyEvent = std::make_shared<detail::event_impl>();

  pi_event PIEvent = nullptr;
  pi_result Res = mock_piEventCreate(/*context = */ (pi_context)0x1, &PIEvent);
  EXPECT_TRUE(PI_SUCCESS == Res);

  auto Event =
      std::make_shared<detail::event_impl>(PIEvent, Queue.get_context());

  using EventList = std::vector<detail::EventImplPtr>;
  std::vector<EventList> InputEventWaitLists = {
      {EmptyEvent}, {Event, Event}, {EmptyEvent, Event}};

  MockScheduler MS;

  for (auto &Arg : InputEventWaitLists) {
    std::unique_ptr<detail::CG> CommandGroup(new detail::CGBarrier(
        {}, detail::CG::StorageInitHelper({}, {}, {}, {}, {std::move(Arg)}),
        detail::CG::CGTYPE::BarrierWaitlist, {}));
    MS.Scheduler::addCG(std::move(CommandGroup), QueueImpl);
  }
}

TEST_F(SchedulerTest, CommandsPiEventExpectation) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  context Ctx{Plt};
  queue Queue{Ctx, default_selector_v};
  detail::QueueImplPtr QueueImpl = detail::getSyclObjImpl(Queue);
  MockScheduler MS;

  buffer<int, 1> Buf{range<1>(1)};
  std::shared_ptr<detail::buffer_impl> BufImpl = detail::getSyclObjImpl(Buf);
  detail::Requirement MockReq = getMockRequirement(Buf);
  MockReq.MDims = 1;
  MockReq.MSYCLMemObj = BufImpl.get();

  std::vector<detail::Command *> AuxCmds;
  detail::MemObjRecord *Record =
      MS.getOrInsertMemObjRecord(QueueImpl, &MockReq, AuxCmds);
  detail::AllocaCommandBase *AllocaCmd =
      MS.getOrCreateAllocaForReq(Record, &MockReq, QueueImpl, AuxCmds);
  EXPECT_EQ(AllocaCmd->producesPiEvent(),
            AllocaCmd->getEvent()->producesPiEvent());
  EXPECT_EQ(AllocaCmd->producesPiEvent(), false);

  std::unique_ptr<detail::CG> CG{
      new detail::CGFill(/*Pattern*/ {}, &MockReq,
                         detail::CG::StorageInitHelper(
                             /*ArgsStorage*/ {},
                             /*AccStorage*/ {},
                             /*SharedPtrStorage*/ {},
                             /*Requirements*/ {&MockReq},
                             /*Events*/ {}))};
  detail::EventImplPtr Event = MS.addCG(std::move(CG), QueueImpl);
  auto *Cmd = static_cast<detail::Command *>(Event->getCommand());
  EXPECT_EQ(Cmd->producesPiEvent(), Event->producesPiEvent());
  EXPECT_EQ(Cmd->producesPiEvent(), true);
}