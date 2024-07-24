//==----- ProfilingTag.cpp --- sycl_ext_oneapi_profiling_tag unit tests ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

template <pi_bool TimestampSupport>
pi_result after_piDeviceGetInfo(pi_device, pi_device_info param_name,
                                size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret) {
  if (param_name == PI_EXT_ONEAPI_DEVICE_INFO_TIMESTAMP_RECORDING_SUPPORT) {
    if (param_value)
      *static_cast<pi_bool *>(param_value) = TimestampSupport;
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(TimestampSupport);
  }
  return PI_SUCCESS;
}

thread_local size_t counter_piEnqueueTimestampRecordingExp = 0;
inline pi_result after_piEnqueueTimestampRecordingExp(pi_queue, pi_bool,
                                                      pi_uint32,
                                                      const pi_event *,
                                                      pi_event *) {
  ++counter_piEnqueueTimestampRecordingExp;
  return PI_SUCCESS;
}

thread_local std::optional<pi_profiling_info> LatestProfilingQuery;
inline pi_result after_piEventGetProfilingInfo(pi_event,
                                               pi_profiling_info param_name,
                                               size_t, void *, size_t *) {
  LatestProfilingQuery = param_name;
  return PI_SUCCESS;
}

thread_local size_t counter_piEnqueueEventsWaitWithBarrier = 0;
inline pi_result after_piEnqueueEventsWaitWithBarrier(pi_queue, pi_uint32,
                                                      const pi_event *,
                                                      pi_event *) {
  ++counter_piEnqueueEventsWaitWithBarrier;
  return PI_SUCCESS;
}

class ProfilingTagTest : public ::testing::Test {
public:
  ProfilingTagTest() : Mock{}, Plt{Mock.getPlatform()} {}

protected:
  void SetUp() override {
    counter_piEnqueueTimestampRecordingExp = 0;
    counter_piEnqueueEventsWaitWithBarrier = 0;
    LatestProfilingQuery = std::nullopt;
  }

protected:
  sycl::unittest::PiMock Mock;
  sycl::platform Plt;
};

TEST_F(ProfilingTagTest, ProfilingTagSupportedDefaultQueue) {
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      after_piDeviceGetInfo<true>);
  Mock.redefineAfter<sycl::detail::PiApiKind::piEnqueueTimestampRecordingExp>(
      after_piEnqueueTimestampRecordingExp);
  Mock.redefineAfter<sycl::detail::PiApiKind::piEventGetProfilingInfo>(
      after_piEventGetProfilingInfo);

  sycl::context Ctx{Plt};
  sycl::queue Queue{Ctx, sycl::default_selector_v};
  sycl::device Dev = Queue.get_device();

  ASSERT_TRUE(Dev.has(sycl::aspect::ext_oneapi_queue_profiling_tag));

  sycl::event E = sycl::ext::oneapi::experimental::submit_profiling_tag(Queue);
  ASSERT_EQ(size_t{1}, counter_piEnqueueTimestampRecordingExp);

  E.get_profiling_info<sycl::info::event_profiling::command_start>();
  ASSERT_TRUE(LatestProfilingQuery.has_value());
  ASSERT_EQ(*LatestProfilingQuery, PI_PROFILING_INFO_COMMAND_START);

  E.get_profiling_info<sycl::info::event_profiling::command_end>();
  ASSERT_TRUE(LatestProfilingQuery.has_value());
  ASSERT_EQ(*LatestProfilingQuery, PI_PROFILING_INFO_COMMAND_END);
}

TEST_F(ProfilingTagTest, ProfilingTagSupportedProfilingQueue) {
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      after_piDeviceGetInfo<true>);
  Mock.redefineAfter<sycl::detail::PiApiKind::piEnqueueTimestampRecordingExp>(
      after_piEnqueueTimestampRecordingExp);
  Mock.redefineAfter<sycl::detail::PiApiKind::piEventGetProfilingInfo>(
      after_piEventGetProfilingInfo);

  sycl::context Ctx{Plt};
  sycl::queue Queue{Ctx,
                    sycl::default_selector_v,
                    {sycl::property::queue::enable_profiling()}};
  sycl::device Dev = Queue.get_device();

  ASSERT_TRUE(Dev.has(sycl::aspect::ext_oneapi_queue_profiling_tag));

  sycl::event E = sycl::ext::oneapi::experimental::submit_profiling_tag(Queue);
  ASSERT_EQ(size_t{1}, counter_piEnqueueTimestampRecordingExp);

  E.get_profiling_info<sycl::info::event_profiling::command_start>();
  ASSERT_TRUE(LatestProfilingQuery.has_value());
  ASSERT_EQ(*LatestProfilingQuery, PI_PROFILING_INFO_COMMAND_START);

  E.get_profiling_info<sycl::info::event_profiling::command_end>();
  ASSERT_TRUE(LatestProfilingQuery.has_value());
  ASSERT_EQ(*LatestProfilingQuery, PI_PROFILING_INFO_COMMAND_END);
}

TEST_F(ProfilingTagTest, ProfilingTagFallbackDefaultQueue) {
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      after_piDeviceGetInfo<false>);
  Mock.redefineAfter<sycl::detail::PiApiKind::piEnqueueTimestampRecordingExp>(
      after_piEnqueueTimestampRecordingExp);
  Mock.redefineAfter<sycl::detail::PiApiKind::piEventGetProfilingInfo>(
      after_piEventGetProfilingInfo);

  sycl::context Ctx{Plt};
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
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      after_piDeviceGetInfo<false>);
  Mock.redefineAfter<sycl::detail::PiApiKind::piEnqueueTimestampRecordingExp>(
      after_piEnqueueTimestampRecordingExp);
  Mock.redefineAfter<sycl::detail::PiApiKind::piEnqueueEventsWaitWithBarrier>(
      after_piEnqueueEventsWaitWithBarrier);

  sycl::context Ctx{Plt};
  sycl::queue Queue{Ctx,
                    sycl::default_selector_v,
                    {sycl::property::queue::enable_profiling()}};
  sycl::device Dev = Queue.get_device();

  ASSERT_FALSE(Dev.has(sycl::aspect::ext_oneapi_queue_profiling_tag));

  sycl::event E = sycl::ext::oneapi::experimental::submit_profiling_tag(Queue);
  ASSERT_EQ(size_t{0}, counter_piEnqueueTimestampRecordingExp);
  ASSERT_EQ(size_t{1}, counter_piEnqueueEventsWaitWithBarrier);
}
