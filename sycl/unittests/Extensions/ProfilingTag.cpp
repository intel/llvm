//==----- ProfilingTag.cpp --- sycl_ext_oneapi_profiling_tag unit tests ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

template <ur_bool_t TimestampSupport>
ur_result_t after_urDeviceGetInfo(void *pParams) {
  auto &Params = *reinterpret_cast<ur_device_get_info_params_t *>(pParams);
  if (*Params.ppropName == UR_DEVICE_INFO_TIMESTAMP_RECORDING_SUPPORT_EXP) {
    if (Params.ppPropValue)
      *static_cast<ur_bool_t *>(*Params.ppPropValue) = TimestampSupport;
    if (*Params.ppPropSizeRet)
      **Params.ppPropSizeRet = sizeof(TimestampSupport);
  }
  return UR_RESULT_SUCCESS;
}

thread_local size_t counter_urEnqueueTimestampRecordingExp = 0;
inline ur_result_t after_urEnqueueTimestampRecordingExp(void *) {
  ++counter_urEnqueueTimestampRecordingExp;
  return UR_RESULT_SUCCESS;
}

thread_local std::optional<ur_profiling_info_t> LatestProfilingQuery;
inline ur_result_t after_urEventGetProfilingInfo(void *pParams) {
  auto &Params =
      *reinterpret_cast<ur_event_get_profiling_info_params_t *>(pParams);
  LatestProfilingQuery = *Params.ppropName;
  return UR_RESULT_SUCCESS;
}

thread_local size_t counter_urEnqueueEventsWaitWithBarrier = 0;
inline ur_result_t after_urEnqueueEventsWaitWithBarrier(void *) {
  ++counter_urEnqueueEventsWaitWithBarrier;
  return UR_RESULT_SUCCESS;
}

class ProfilingTagTest : public ::testing::Test {
public:
  ProfilingTagTest() : Mock{} {}

protected:
  void SetUp() override {
    counter_urEnqueueTimestampRecordingExp = 0;
    counter_urEnqueueEventsWaitWithBarrier = 0;
    LatestProfilingQuery = std::nullopt;
  }

protected:
  sycl::unittest::UrMock<> Mock;
};

TEST_F(ProfilingTagTest, ProfilingTagSupportedDefaultQueue) {
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &after_urDeviceGetInfo<true>);
  mock::getCallbacks().set_after_callback(
      "urEnqueueTimestampRecordingExp", &after_urEnqueueTimestampRecordingExp);
  mock::getCallbacks().set_after_callback("urEventGetProfilingInfo",
                                          &after_urEventGetProfilingInfo);
  mock::getCallbacks().set_after_callback(
      "urEnqueueEventsWaitWithBarrier", &after_urEnqueueEventsWaitWithBarrier);

  sycl::context Ctx{sycl::platform()};
  sycl::queue Queue{Ctx, sycl::default_selector_v};
  sycl::device Dev = Queue.get_device();

  ASSERT_TRUE(Dev.has(sycl::aspect::ext_oneapi_queue_profiling_tag));

  sycl::event E = sycl::ext::oneapi::experimental::submit_profiling_tag(Queue);
  ASSERT_EQ(size_t{1}, counter_urEnqueueTimestampRecordingExp);
  // TODO: We expect two barriers for now, while marker events leak. Adjust when
  //       addressed.
  ASSERT_EQ(size_t{2}, counter_urEnqueueEventsWaitWithBarrier);

  E.get_profiling_info<sycl::info::event_profiling::command_start>();
  ASSERT_TRUE(LatestProfilingQuery.has_value());
  ASSERT_EQ(*LatestProfilingQuery, UR_PROFILING_INFO_COMMAND_START);

  E.get_profiling_info<sycl::info::event_profiling::command_end>();
  ASSERT_TRUE(LatestProfilingQuery.has_value());
  ASSERT_EQ(*LatestProfilingQuery, UR_PROFILING_INFO_COMMAND_END);
}

TEST_F(ProfilingTagTest, ProfilingTagSupportedInOrderQueue) {
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &after_urDeviceGetInfo<true>);
  mock::getCallbacks().set_after_callback(
      "urEnqueueTimestampRecordingExp", &after_urEnqueueTimestampRecordingExp);
  mock::getCallbacks().set_after_callback("urEventGetProfilingInfo",
                                          &after_urEventGetProfilingInfo);
  mock::getCallbacks().set_after_callback(
      "urEnqueueEventsWaitWithBarrier", &after_urEnqueueEventsWaitWithBarrier);

  sycl::context Ctx{sycl::platform()};
  sycl::queue Queue{
      Ctx, sycl::default_selector_v, {sycl::property::queue::in_order()}};
  sycl::device Dev = Queue.get_device();

  ASSERT_TRUE(Dev.has(sycl::aspect::ext_oneapi_queue_profiling_tag));

  sycl::event E = sycl::ext::oneapi::experimental::submit_profiling_tag(Queue);
  ASSERT_EQ(size_t{1}, counter_urEnqueueTimestampRecordingExp);
  ASSERT_EQ(size_t{0}, counter_urEnqueueEventsWaitWithBarrier);

  E.get_profiling_info<sycl::info::event_profiling::command_start>();
  ASSERT_TRUE(LatestProfilingQuery.has_value());
  ASSERT_EQ(*LatestProfilingQuery, UR_PROFILING_INFO_COMMAND_START);

  E.get_profiling_info<sycl::info::event_profiling::command_end>();
  ASSERT_TRUE(LatestProfilingQuery.has_value());
  ASSERT_EQ(*LatestProfilingQuery, UR_PROFILING_INFO_COMMAND_END);
}

TEST_F(ProfilingTagTest, ProfilingTagSupportedProfilingQueue) {
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &after_urDeviceGetInfo<true>);
  mock::getCallbacks().set_after_callback(
      "urEnqueueTimestampRecordingExp", &after_urEnqueueTimestampRecordingExp);
  mock::getCallbacks().set_after_callback("urEventGetProfilingInfo",
                                          &after_urEventGetProfilingInfo);

  sycl::context Ctx{sycl::platform()};
  sycl::queue Queue{Ctx,
                    sycl::default_selector_v,
                    {sycl::property::queue::enable_profiling()}};
  sycl::device Dev = Queue.get_device();

  ASSERT_TRUE(Dev.has(sycl::aspect::ext_oneapi_queue_profiling_tag));

  sycl::event E = sycl::ext::oneapi::experimental::submit_profiling_tag(Queue);
  ASSERT_EQ(size_t{1}, counter_urEnqueueTimestampRecordingExp);

  E.get_profiling_info<sycl::info::event_profiling::command_start>();
  ASSERT_TRUE(LatestProfilingQuery.has_value());
  ASSERT_EQ(*LatestProfilingQuery, UR_PROFILING_INFO_COMMAND_START);

  E.get_profiling_info<sycl::info::event_profiling::command_end>();
  ASSERT_TRUE(LatestProfilingQuery.has_value());
  ASSERT_EQ(*LatestProfilingQuery, UR_PROFILING_INFO_COMMAND_END);
}

TEST_F(ProfilingTagTest, ProfilingTagSupportedProfilingInOrderQueue) {
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &after_urDeviceGetInfo<true>);
  mock::getCallbacks().set_after_callback(
      "urEnqueueTimestampRecordingExp", &after_urEnqueueTimestampRecordingExp);
  mock::getCallbacks().set_after_callback("urEventGetProfilingInfo",
                                          &after_urEventGetProfilingInfo);
  mock::getCallbacks().set_after_callback(
      "urEnqueueEventsWaitWithBarrier", &after_urEnqueueEventsWaitWithBarrier);

  sycl::context Ctx{sycl::platform()};
  sycl::queue Queue{Ctx,
                    sycl::default_selector_v,
                    {sycl::property::queue::enable_profiling(),
                     sycl::property::queue::in_order()}};
  sycl::device Dev = Queue.get_device();

  ASSERT_TRUE(Dev.has(sycl::aspect::ext_oneapi_queue_profiling_tag));

  sycl::event E = sycl::ext::oneapi::experimental::submit_profiling_tag(Queue);
  ASSERT_EQ(size_t{1}, counter_urEnqueueTimestampRecordingExp);
  ASSERT_EQ(size_t{0}, counter_urEnqueueEventsWaitWithBarrier);

  E.get_profiling_info<sycl::info::event_profiling::command_start>();
  ASSERT_TRUE(LatestProfilingQuery.has_value());
  ASSERT_EQ(*LatestProfilingQuery, UR_PROFILING_INFO_COMMAND_START);

  E.get_profiling_info<sycl::info::event_profiling::command_end>();
  ASSERT_TRUE(LatestProfilingQuery.has_value());
  ASSERT_EQ(*LatestProfilingQuery, UR_PROFILING_INFO_COMMAND_END);
}

TEST_F(ProfilingTagTest, ProfilingTagFallbackDefaultQueue) {
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &after_urDeviceGetInfo<false>);
  mock::getCallbacks().set_after_callback(
      "urEnqueueTimestampRecordingExp", &after_urEnqueueTimestampRecordingExp);
  mock::getCallbacks().set_after_callback("urEventGetProfilingInfo",
                                          &after_urEventGetProfilingInfo);

  sycl::context Ctx{sycl::platform()};
  sycl::queue Queue{Ctx, sycl::default_selector_v};
  sycl::device Dev = Queue.get_device();

  ASSERT_FALSE(Dev.has(sycl::aspect::ext_oneapi_queue_profiling_tag));

  try {
    sycl::ext::oneapi::experimental::submit_profiling_tag(Queue);
    FAIL() << "Exception was not thrown.";
  } catch (sycl::exception &E) {
    ASSERT_EQ(E.code(), sycl::make_error_code(sycl::errc::invalid));
  }
}

TEST_F(ProfilingTagTest, ProfilingTagFallbackProfilingQueue) {
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &after_urDeviceGetInfo<false>);
  mock::getCallbacks().set_after_callback(
      "urEnqueueTimestampRecordingExp", &after_urEnqueueTimestampRecordingExp);
  mock::getCallbacks().set_after_callback(
      "urEnqueueEventsWaitWithBarrier", &after_urEnqueueEventsWaitWithBarrier);

  sycl::context Ctx{sycl::platform()};
  sycl::queue Queue{Ctx,
                    sycl::default_selector_v,
                    {sycl::property::queue::enable_profiling()}};
  sycl::device Dev = Queue.get_device();

  ASSERT_FALSE(Dev.has(sycl::aspect::ext_oneapi_queue_profiling_tag));

  sycl::event E = sycl::ext::oneapi::experimental::submit_profiling_tag(Queue);
  ASSERT_EQ(size_t{0}, counter_urEnqueueTimestampRecordingExp);
  ASSERT_EQ(size_t{1}, counter_urEnqueueEventsWaitWithBarrier);
}
