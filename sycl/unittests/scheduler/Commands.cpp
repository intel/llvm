//==----------- Commands.cpp --- Commands unit tests -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"
#include "ur_mock_helpers.hpp"
#include <helpers/UrMock.hpp>

#include <iostream>

using namespace sycl;

ur_result_t redefineEnqueueEventsWaitWithBarrier(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_events_wait_with_barrier_params_t *>(pParams);

  for (uint32_t i = 0; i != *params.pnumEventsInWaitList; ++i)
    EXPECT_NE((*params.pphEventWaitList)[i], nullptr);

  return UR_RESULT_SUCCESS;
}

// Hack that allows to return a context in redefinePiEventGetInfo
ur_context_handle_t queue_global_context = nullptr;

ur_result_t redefineUrEventGetInfo(void *pParams) {
  auto params = *static_cast<ur_event_get_info_params_t *>(pParams);
  *reinterpret_cast<ur_context_handle_t *>(*params.ppPropValue) =
      queue_global_context;
  return UR_RESULT_SUCCESS;
}

//
// This test checks a handling of empty events in WaitWithBarrier command.
// Original reproducer for l0 adapter led to segfault(nullptr dereference):
//
// #include <sycl/sycl.hpp>
// int main() {
//     sycl::queue q;
//     sycl::event e;
//     q.submit_barrier({e});
// }
//
TEST_F(SchedulerTest, WaitEmptyEventWithBarrier) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  mock::getCallbacks().set_before_callback(
      "urEnqueueEventsWaitWithBarrier", &redefineEnqueueEventsWaitWithBarrier);

  queue Queue{Plt.get_devices()[0]};
  sycl::detail::QueueImplPtr QueueImpl = detail::getSyclObjImpl(Queue);

  queue_global_context =
      detail::getSyclObjImpl(Queue.get_context())->getHandleRef();

  mock::getCallbacks().set_before_callback("urEventGetInfo",
                                           &redefineUrEventGetInfo);

  auto EmptyEvent = std::make_shared<detail::event_impl>();

  ur_event_handle_t UREvent = mock::createDummyHandle<ur_event_handle_t>();

  auto Event =
      std::make_shared<detail::event_impl>(UREvent, Queue.get_context());

  using EventList = std::vector<detail::EventImplPtr>;
  std::vector<EventList> InputEventWaitLists = {
      {EmptyEvent}, {Event, Event}, {EmptyEvent, Event}};

  MockScheduler MS;

  for (auto &Arg : InputEventWaitLists) {
    std::unique_ptr<detail::CG> CommandGroup(new detail::CGBarrier(
        std::move(Arg), detail::CG::StorageInitHelper({}, {}, {}, {}, {}),
        detail::CGType::BarrierWaitlist, {}));
    MS.Scheduler::addCG(std::move(CommandGroup), QueueImpl,
                        /*EventNeeded=*/true);
  }
}
