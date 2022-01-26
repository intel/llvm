//==----------- Commands.cpp --- Commands unit tests -----------------------==//
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

using namespace cl::sycl;

pi_result redefinePiEnqueueEventsWaitWithBarrier(pi_queue Queue,
                                                 pi_uint32 NumEventsInWaitList,
                                                 const pi_event *EventWaitList,
                                                 pi_event *Event) {

  for (pi_uint32 i = 0; i != NumEventsInWaitList; ++i)
    EXPECT_NE(EventWaitList[i], nullptr);

  return PI_SUCCESS;
}

// Hack that allows to return a context in redefinePiEventGetInfo
RT::PiContext queue_global_context = nullptr;

pi_result redefinePiEventGetInfo(pi_event, pi_event_info, size_t,
                                 void *param_value, size_t *) {
  *reinterpret_cast<RT::PiContext *>(param_value) = queue_global_context;
  return PI_SUCCESS;
}

pi_result redefinePiEventRetain(pi_event) { return PI_SUCCESS; }

pi_result redefinePiEventRelease(pi_event) { return PI_SUCCESS; }

//
// This test checks a handling of empty events in WaitWithBarrier command.
// Original reproducer for l0 plugin led to segfault(nullptr dereference):
//
// #include <CL/sycl.hpp>
// int main() {
//     sycl::queue q;
//     sycl::event e;
//     q.submit_barrier({e});
// }
//
TEST_F(SchedulerTest, WaitEmptyEventWithBarrier) {
  // NB! This test requires at least one non-host environmet
  // For example, OpenCL.
  default_selector Selector{};
  if (Selector.select_device().is_host()) {
    std::cerr << "Not run due to host-only environment\n";
    return;
  }

  platform Plt{Selector};
  unittest::PiMock Mock{Plt};

  Mock.redefine<detail::PiApiKind::piEnqueueEventsWaitWithBarrier>(
      redefinePiEnqueueEventsWaitWithBarrier);

  queue Queue{Plt.get_devices()[0]};
  cl::sycl::detail::QueueImplPtr QueueImpl = detail::getSyclObjImpl(Queue);

  queue_global_context =
      detail::getSyclObjImpl(Queue.get_context())->getHandleRef();

  Mock.redefine<detail::PiApiKind::piEventGetInfo>(redefinePiEventGetInfo);
  Mock.redefine<detail::PiApiKind::piEventRetain>(redefinePiEventRetain);
  Mock.redefine<detail::PiApiKind::piEventRelease>(redefinePiEventRelease);

  auto EmptyEvent = std::make_shared<detail::event_impl>();
  auto Event = std::make_shared<detail::event_impl>(
      reinterpret_cast<RT::PiEvent>(0x01), Queue.get_context());

  using EventList = std::vector<detail::EventImplPtr>;
  std::vector<EventList> InputEventWaitLists = {
      {EmptyEvent}, {Event, Event}, {EmptyEvent, Event}};

  MockScheduler MS;

  for (auto &Arg : InputEventWaitLists) {
    std::unique_ptr<detail::CG> CommandGroup(
        new detail::CGBarrier(std::move(Arg), {}, {}, {}, {}, {},
                              detail::CG::CGTYPE::BarrierWaitlist, {}));
    MS.Scheduler::addCG(std::move(CommandGroup), QueueImpl);
  }
}
