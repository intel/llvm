//==------ ProfilingTag.cpp --- sycl_ext_oneapi_event_mode unit tests ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <utility>

inline thread_local size_t counter_urEnqueueEventsWaitWithBarrierExt = 0;
inline ur_result_t after_urEnqueueEventsWaitWithBarrierExt(void *pParams) {
  auto Params =
      *static_cast<ur_enqueue_events_wait_with_barrier_ext_params_t *>(pParams);
  std::ignore = Params;

  assert(*Params.ppProperties != nullptr);
  assert((*Params.ppProperties)->flags &
         UR_EXP_ENQUEUE_EXT_FLAG_LOW_POWER_EVENTS);

  ++counter_urEnqueueEventsWaitWithBarrierExt;
  return UR_RESULT_SUCCESS;
}

class EventModeTest : public ::testing::Test {
public:
  EventModeTest() : Mock{} {}

protected:
  void SetUp() override { counter_urEnqueueEventsWaitWithBarrierExt = 0; }

protected:
  sycl::unittest::UrMock<> Mock;
};

TEST_F(EventModeTest, EventModeFullBarrier) {
  mock::getCallbacks().set_after_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &after_urEnqueueEventsWaitWithBarrierExt);

  sycl::queue Q;

  sycl::ext::oneapi::experimental::properties Props{
      sycl::ext::oneapi::experimental::event_mode{
          sycl::ext::oneapi::experimental::event_mode_enum::low_power}};

  sycl::ext::oneapi::experimental::submit_with_event(
      Q, Props,
      [&](sycl::handler &CGH) {
        sycl::ext::oneapi::experimental::barrier(CGH);
      })
      .wait();

  ASSERT_EQ(size_t{1}, counter_urEnqueueEventsWaitWithBarrierExt);
}

TEST_F(EventModeTest, EventModePartialBarrier) {
  mock::getCallbacks().set_after_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &after_urEnqueueEventsWaitWithBarrierExt);

  sycl::queue Q;

  sycl::ext::oneapi::experimental::properties Props{
      sycl::ext::oneapi::experimental::event_mode{
          sycl::ext::oneapi::experimental::event_mode_enum::low_power}};

  sycl::event E = Q.prefetch(reinterpret_cast<void *>(0x1), 1);

  sycl::ext::oneapi::experimental::submit_with_event(
      Q, Props,
      [&](sycl::handler &CGH) {
        sycl::ext::oneapi::experimental::partial_barrier(CGH, {E});
      })
      .wait();

  ASSERT_EQ(size_t{1}, counter_urEnqueueEventsWaitWithBarrierExt);
}

TEST_F(EventModeTest, EventModeInOrderFullBarrier) {
  mock::getCallbacks().set_after_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &after_urEnqueueEventsWaitWithBarrierExt);

  sycl::queue Q{sycl::property::queue::in_order{}};

  sycl::ext::oneapi::experimental::properties Props{
      sycl::ext::oneapi::experimental::event_mode{
          sycl::ext::oneapi::experimental::event_mode_enum::low_power}};

  Q.prefetch(reinterpret_cast<void *>(0x1), 1);

  sycl::ext::oneapi::experimental::submit_with_event(
      Q, Props,
      [&](sycl::handler &CGH) {
        sycl::ext::oneapi::experimental::barrier(CGH);
      })
      .wait();

  ASSERT_EQ(size_t{1}, counter_urEnqueueEventsWaitWithBarrierExt);
}

TEST_F(EventModeTest, EventModeInOrderPartialBarrier) {
  mock::getCallbacks().set_after_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &after_urEnqueueEventsWaitWithBarrierExt);

  sycl::queue Q{sycl::property::queue::in_order{}};

  sycl::ext::oneapi::experimental::properties Props{
      sycl::ext::oneapi::experimental::event_mode{
          sycl::ext::oneapi::experimental::event_mode_enum::low_power}};

  Q.prefetch(reinterpret_cast<void *>(0x1), 1);

  sycl::event E = Q.prefetch(reinterpret_cast<void *>(0x1), 1);

  sycl::ext::oneapi::experimental::submit_with_event(
      Q, Props,
      [&](sycl::handler &CGH) {
        sycl::ext::oneapi::experimental::partial_barrier(CGH, {E});
      })
      .wait();

  ASSERT_EQ(size_t{1}, counter_urEnqueueEventsWaitWithBarrierExt);
}
