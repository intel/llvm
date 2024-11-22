//==------------------------- GetLastEvent.cpp -----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Tests the behavior of queue::ext_oneapi_get_last_event.

#include <detail/event_impl.hpp>
#include <gtest/gtest.h>
#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/queue.hpp>

using namespace sycl;

thread_local ur_event_handle_t MarkerEventLatest = nullptr;
static ur_result_t redefinedEnqueueEventsWaitAfter(void *pParams) {
  auto params = *static_cast<ur_enqueue_events_wait_params_t *>(pParams);
  MarkerEventLatest = **(params.pphEvent);
  return UR_RESULT_SUCCESS;
}
static ur_result_t redefinedEventRelease(void *) { return UR_RESULT_SUCCESS; }

TEST(GetLastEventEmptyQueue, CheckEmptyQueueLastEvent) {
  unittest::UrMock<> Mock;
  platform Plt = sycl::platform();

  MarkerEventLatest = nullptr;
  mock::getCallbacks().set_after_callback("urEnqueueEventsWait",
                                          &redefinedEnqueueEventsWaitAfter);
  mock::getCallbacks().set_before_callback("urEventRelease",
                                           &redefinedEventRelease);

  queue Q{property::queue::in_order{}};
  event E = Q.ext_oneapi_get_last_event();
  ur_event_handle_t UREvent = detail::getSyclObjImpl(E)->getHandle();
  ASSERT_NE(MarkerEventLatest, ur_event_handle_t{nullptr});
  ASSERT_EQ(UREvent, MarkerEventLatest);
}

TEST(GetLastEventEmptyQueue, CheckEventlessWorkQueue) {
  unittest::UrMock<> Mock;
  platform Plt = sycl::platform();

  MarkerEventLatest = nullptr;
  mock::getCallbacks().set_after_callback("urEnqueueEventsWait",
                                          &redefinedEnqueueEventsWaitAfter);
  mock::getCallbacks().set_before_callback("urEventRelease",
                                           &redefinedEventRelease);

  queue Q{property::queue::in_order{}};

  // The following single_task does not return an event, so it is expected that
  // the last event query creates a new marker event.
  sycl::ext::oneapi::experimental::single_task<TestKernel<>>(Q, []() {});
  event E = Q.ext_oneapi_get_last_event();
  ur_event_handle_t UREvent = detail::getSyclObjImpl(E)->getHandle();
  ASSERT_NE(MarkerEventLatest, ur_event_handle_t{nullptr});
  ASSERT_EQ(UREvent, MarkerEventLatest);
}
