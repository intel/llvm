//==------------------- ExtOneapiBarrierOpt.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/event_impl.hpp>
#include <gtest/gtest.h>
#include <helpers/ScopedEnvVar.hpp>
#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

inline thread_local uint32_t NumEventsInWaitList;

static ur_result_t redefinedEnqueueEventsWaitWithBarrierExt(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_events_wait_with_barrier_ext_params_t *>(pParams);
  NumEventsInWaitList = *(params.pnumEventsInWaitList);
  return UR_RESULT_SUCCESS;
}

class ExtOneapiBarrierOptTest : public ::testing::Test {
public:
  ExtOneapiBarrierOptTest() : Mock{} {}

protected:
  void SetUp() override { NumEventsInWaitList = 0; }

protected:
  sycl::unittest::UrMock<> Mock;
};

// Check that ext_oneapi_submit_barrier works fine in the scenarios
// when provided waitlist consists of only empty events.
// Tets for https://github.com/intel/llvm/pull/12951
TEST_F(ExtOneapiBarrierOptTest, EmptyEventTest) {
  sycl::queue q1{{sycl::property::queue::in_order()}};

  mock::getCallbacks().set_after_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefinedEnqueueEventsWaitWithBarrierExt);

  NumEventsInWaitList = 100;
  q1.ext_oneapi_submit_barrier();
  ASSERT_EQ(0u, NumEventsInWaitList);

  // ext_oneapi_submit_barrier should ignore empty, default constructed events.
  // Spec says that ext_oneapi_submit_barrier({E1}) should submit a barrier
  // that's blocked on E1. But if E1 is empty event, SYCL RT will ignore it and
  // won't make the UR call at all.
  sycl::event E1{};
  NumEventsInWaitList = 100;
  q1.ext_oneapi_submit_barrier({E1});
  ASSERT_EQ(100u, NumEventsInWaitList);
}

// A barrier with a wait list which is filtered out as redundant still has to
// reach the backend when its event is needed. An in-order queue keeps that
// event as the dependency of the commands submitted after it, and an event
// without a backend handle is read as a command which has not been submitted to
// the backend yet, see Scheduler::areEventsSafeForSchedulerBypass.
TEST_F(ExtOneapiBarrierOptTest, RedundantWaitListStillProducesEvent) {
  sycl::queue q{{sycl::property::queue::in_order()}};

  mock::getCallbacks().set_after_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefinedEnqueueEventsWaitWithBarrierExt);

  // A command on this very queue. A barrier submitted to an in-order queue does
  // not have to wait for its event again, so the wait list comes out empty.
  sycl::event E = q.submit(
      [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });

  // The handler based barrier is always handled by the scheduler.
  NumEventsInWaitList = 100;
  sycl::event Barrier =
      q.submit([&](sycl::handler &CGH) { CGH.ext_oneapi_barrier({E}); });

  detail::event_impl &BarrierImpl = *detail::getSyclObjImpl(Barrier);
  EXPECT_EQ(0u, NumEventsInWaitList);
  EXPECT_NE(BarrierImpl.getHandle(), nullptr);

  q.wait();
}
