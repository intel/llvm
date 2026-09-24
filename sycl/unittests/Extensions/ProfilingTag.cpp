//==----- ProfilingTag.cpp --- sycl_ext_oneapi_profiling_tag unit tests ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <detail/event_impl.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <tuple>

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

// Simulates a backend that cannot record a device timestamp (e.g. the OpenCL
// backend on a queue without profiling enabled), so that the scheduler falls
// back to a barrier.
inline ur_result_t replace_urEnqueueTimestampRecordingExpUnsupported(void *) {
  ++counter_urEnqueueTimestampRecordingExp;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

inline ur_result_t replace_urEnqueueTimestampRecordingExpFailure(void *) {
  ++counter_urEnqueueTimestampRecordingExp;
  return UR_RESULT_ERROR_UNKNOWN;
}

thread_local std::optional<ur_profiling_info_t> LatestProfilingQuery;
inline ur_result_t after_urEventGetProfilingInfo(void *pParams) {
  auto &Params =
      *reinterpret_cast<ur_event_get_profiling_info_params_t *>(pParams);
  LatestProfilingQuery = *Params.ppropName;
  return UR_RESULT_SUCCESS;
}

constexpr uint64_t ProfilingTagSubmitTime = 11;
constexpr uint64_t ProfilingTagStartTime = 21;
constexpr uint64_t ProfilingTagEndTime = 42;
inline thread_local size_t ProfilingTagGlobalTimestampQueries = 0;
inline thread_local ur_event_handle_t LastWaitedProfilingTagEvent = nullptr;

inline ur_result_t
replace_urDeviceGetGlobalTimestampsForProfilingTag(void *pParams) {
  auto &Params =
      *static_cast<ur_device_get_global_timestamps_params_t *>(pParams);
  ++ProfilingTagGlobalTimestampQueries;
  if (*Params.ppDeviceTimestamp)
    **Params.ppDeviceTimestamp = ProfilingTagSubmitTime;
  if (*Params.ppHostTimestamp)
    **Params.ppHostTimestamp = ProfilingTagSubmitTime;
  return UR_RESULT_SUCCESS;
}

inline ur_result_t after_urEventWaitForProfilingTag(void *pParams) {
  auto &Params = *static_cast<ur_event_wait_params_t *>(pParams);
  EXPECT_EQ(*Params.pnumEvents, 1u);
  LastWaitedProfilingTagEvent = **Params.pphEventWaitList;
  return UR_RESULT_SUCCESS;
}

inline ur_result_t
replace_urEventGetProfilingInfoForProfilingTag(void *pParams) {
  auto &Params = *static_cast<ur_event_get_profiling_info_params_t *>(pParams);
  LatestProfilingQuery = *Params.ppropName;
  // Native CPU exposes start/end times, but not command_submit.
  if (*Params.ppropName == UR_PROFILING_INFO_COMMAND_SUBMIT)
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  if (*Params.ppropName != UR_PROFILING_INFO_COMMAND_START &&
      *Params.ppropName != UR_PROFILING_INFO_COMMAND_END)
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  // A GPU-written timestamp is not available until the event completes.
  if (*Params.phEvent != LastWaitedProfilingTagEvent)
    return UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE;
  if (*Params.ppPropValue)
    *static_cast<uint64_t *>(*Params.ppPropValue) =
        *Params.ppropName == UR_PROFILING_INFO_COMMAND_START
            ? ProfilingTagStartTime
            : ProfilingTagEndTime;
  if (*Params.ppPropSizeRet)
    **Params.ppPropSizeRet = sizeof(uint64_t);
  return UR_RESULT_SUCCESS;
}

inline thread_local size_t counter_urEnqueueEventsWaitWithBarrierExt = 0;
inline thread_local ur_event_handle_t LatestBarrierEvent = nullptr;
inline thread_local bool LatestBarrierEventReleased = false;
inline ur_result_t after_urEnqueueEventsWaitWithBarrierExt(void *pParams) {
  ++counter_urEnqueueEventsWaitWithBarrierExt;
  auto &Params =
      *static_cast<ur_enqueue_events_wait_with_barrier_params_t *>(pParams);
  if (*Params.pphEvent)
    LatestBarrierEvent = **Params.pphEvent;
  return UR_RESULT_SUCCESS;
}

inline ur_result_t before_urEventRelease(void *pParams) {
  auto &Params = *static_cast<ur_event_release_params_t *>(pParams);
  if (LatestBarrierEvent && *Params.phEvent == LatestBarrierEvent)
    LatestBarrierEventReleased = true;
  return UR_RESULT_SUCCESS;
}

class ProfilingTagTest : public ::testing::Test {
public:
  ProfilingTagTest() : Mock{} {}

protected:
  void SetUp() override {
    counter_urEnqueueTimestampRecordingExp = 0;
    counter_urEnqueueEventsWaitWithBarrierExt = 0;
    LatestBarrierEvent = nullptr;
    LatestBarrierEventReleased = false;
    LatestProfilingQuery = std::nullopt;
    ProfilingTagGlobalTimestampQueries = 0;
    LastWaitedProfilingTagEvent = nullptr;
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
      "urEnqueueEventsWaitWithBarrier",
      &after_urEnqueueEventsWaitWithBarrierExt);

  sycl::context Ctx{sycl::platform()};
  sycl::queue Queue{Ctx, sycl::default_selector_v};
  sycl::device Dev = Queue.get_device();

  ASSERT_TRUE(Dev.has(sycl::aspect::ext_oneapi_queue_profiling_tag));

  sycl::event E = sycl::ext::oneapi::experimental::submit_profiling_tag(Queue);
  ASSERT_EQ(size_t{1}, counter_urEnqueueTimestampRecordingExp);
  // TODO: We expect two barriers for now, while marker events leak. Adjust when
  //       addressed.
  ASSERT_EQ(size_t{2}, counter_urEnqueueEventsWaitWithBarrierExt);

  E.get_profiling_info<sycl::info::event_profiling::command_submit>();
  ASSERT_TRUE(LatestProfilingQuery.has_value());
  ASSERT_EQ(*LatestProfilingQuery, UR_PROFILING_INFO_COMMAND_END);

  E.get_profiling_info<sycl::info::event_profiling::command_start>();
  ASSERT_TRUE(LatestProfilingQuery.has_value());
  ASSERT_EQ(*LatestProfilingQuery, UR_PROFILING_INFO_COMMAND_END);

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
      "urEnqueueEventsWaitWithBarrier",
      &after_urEnqueueEventsWaitWithBarrierExt);

  sycl::context Ctx{sycl::platform()};
  sycl::queue Queue{
      Ctx, sycl::default_selector_v, {sycl::property::queue::in_order()}};
  sycl::device Dev = Queue.get_device();

  ASSERT_TRUE(Dev.has(sycl::aspect::ext_oneapi_queue_profiling_tag));

  sycl::event E = sycl::ext::oneapi::experimental::submit_profiling_tag(Queue);
  ASSERT_EQ(size_t{1}, counter_urEnqueueTimestampRecordingExp);
  ASSERT_EQ(size_t{0}, counter_urEnqueueEventsWaitWithBarrierExt);

  E.get_profiling_info<sycl::info::event_profiling::command_start>();
  ASSERT_TRUE(LatestProfilingQuery.has_value());
  ASSERT_EQ(*LatestProfilingQuery, UR_PROFILING_INFO_COMMAND_END);

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
  ASSERT_EQ(*LatestProfilingQuery, UR_PROFILING_INFO_COMMAND_END);

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
      "urEnqueueEventsWaitWithBarrier",
      &after_urEnqueueEventsWaitWithBarrierExt);

  sycl::context Ctx{sycl::platform()};
  sycl::queue Queue{Ctx,
                    sycl::default_selector_v,
                    {sycl::property::queue::enable_profiling(),
                     sycl::property::queue::in_order()}};
  sycl::device Dev = Queue.get_device();

  ASSERT_TRUE(Dev.has(sycl::aspect::ext_oneapi_queue_profiling_tag));

  sycl::event E = sycl::ext::oneapi::experimental::submit_profiling_tag(Queue);
  ASSERT_EQ(size_t{1}, counter_urEnqueueTimestampRecordingExp);
  ASSERT_EQ(size_t{0}, counter_urEnqueueEventsWaitWithBarrierExt);

  E.get_profiling_info<sycl::info::event_profiling::command_start>();
  ASSERT_TRUE(LatestProfilingQuery.has_value());
  ASSERT_EQ(*LatestProfilingQuery, UR_PROFILING_INFO_COMMAND_END);

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

// Without the aspect but with profiling enabled, the tag is still serviced via
// the profiling-tag command group. If the backend can record a device
// timestamp (as the OpenCL backend now does on a profiling-enabled queue), the
// native recording path is used rather than a bare barrier.
TEST_F(ProfilingTagTest, ProfilingTagFallbackProfilingQueueTimestamp) {
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &after_urDeviceGetInfo<false>);
  mock::getCallbacks().set_after_callback(
      "urEnqueueTimestampRecordingExp", &after_urEnqueueTimestampRecordingExp);
  mock::getCallbacks().set_after_callback(
      "urEnqueueEventsWaitWithBarrier",
      &after_urEnqueueEventsWaitWithBarrierExt);

  sycl::context Ctx{sycl::platform()};
  sycl::queue Queue{Ctx,
                    sycl::default_selector_v,
                    {sycl::property::queue::enable_profiling(),
                     sycl::property::queue::in_order()}};
  sycl::device Dev = Queue.get_device();

  ASSERT_FALSE(Dev.has(sycl::aspect::ext_oneapi_queue_profiling_tag));

  sycl::event E = sycl::ext::oneapi::experimental::submit_profiling_tag(Queue);
  ASSERT_EQ(size_t{1}, counter_urEnqueueTimestampRecordingExp);
  ASSERT_EQ(size_t{0}, counter_urEnqueueEventsWaitWithBarrierExt);
}

// If the backend reports that it cannot record a device timestamp, the
// scheduler must gracefully fall back to a barrier.
TEST_F(ProfilingTagTest, ProfilingTagFallbackProfilingQueueBarrier) {
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &after_urDeviceGetInfo<false>);
  mock::getCallbacks().set_replace_callback(
      "urEnqueueTimestampRecordingExp",
      &replace_urEnqueueTimestampRecordingExpUnsupported);
  mock::getCallbacks().set_after_callback(
      "urEnqueueEventsWaitWithBarrier",
      &after_urEnqueueEventsWaitWithBarrierExt);

  sycl::context Ctx{sycl::platform()};
  sycl::queue Queue{Ctx,
                    sycl::default_selector_v,
                    {sycl::property::queue::enable_profiling(),
                     sycl::property::queue::in_order()}};
  sycl::device Dev = Queue.get_device();

  ASSERT_FALSE(Dev.has(sycl::aspect::ext_oneapi_queue_profiling_tag));

  sycl::event E = sycl::ext::oneapi::experimental::submit_profiling_tag(Queue);
  // The timestamp recording is attempted and reports unsupported, so a single
  // barrier is used as the fallback on this in-order queue.
  ASSERT_EQ(size_t{1}, counter_urEnqueueTimestampRecordingExp);
  ASSERT_EQ(size_t{1}, counter_urEnqueueEventsWaitWithBarrierExt);
}

TEST_F(ProfilingTagTest,
       ProfilingTagFallbackProfilingOutOfOrderQueueReusesBarrier) {
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &after_urDeviceGetInfo<false>);
  mock::getCallbacks().set_replace_callback(
      "urEnqueueTimestampRecordingExp",
      &replace_urEnqueueTimestampRecordingExpUnsupported);
  mock::getCallbacks().set_after_callback(
      "urEnqueueEventsWaitWithBarrier",
      &after_urEnqueueEventsWaitWithBarrierExt);

  sycl::context Ctx{sycl::platform()};
  sycl::queue Queue{Ctx,
                    sycl::default_selector_v,
                    {sycl::property::queue::enable_profiling()}};
  sycl::device Dev = Queue.get_device();

  ASSERT_FALSE(Dev.has(sycl::aspect::ext_oneapi_queue_profiling_tag));

  sycl::event E = sycl::ext::oneapi::experimental::submit_profiling_tag(Queue);
  ASSERT_EQ(size_t{1}, counter_urEnqueueTimestampRecordingExp);
  ASSERT_EQ(size_t{1}, counter_urEnqueueEventsWaitWithBarrierExt);
  ASSERT_NE(nullptr, LatestBarrierEvent);
  ASSERT_EQ(LatestBarrierEvent, sycl::detail::getSyclObjImpl(E)->getHandle());
}

class ProfilingTagTimestampTest
    : public ProfilingTagTest,
      public ::testing::WithParamInterface<std::tuple<bool, bool>> {};

TEST_P(ProfilingTagTimestampTest, UsesCompletionTimeWithoutExplicitWait) {
  const auto [TimestampSupported, IsInOrder] = GetParam();
  mock::getCallbacks().set_after_callback(
      "urDeviceGetInfo", TimestampSupported ? &after_urDeviceGetInfo<true>
                                            : &after_urDeviceGetInfo<false>);
  if (TimestampSupported)
    mock::getCallbacks().set_after_callback(
        "urEnqueueTimestampRecordingExp",
        &after_urEnqueueTimestampRecordingExp);
  else
    mock::getCallbacks().set_replace_callback(
        "urEnqueueTimestampRecordingExp",
        &replace_urEnqueueTimestampRecordingExpUnsupported);
  mock::getCallbacks().set_replace_callback(
      "urDeviceGetGlobalTimestamps",
      &replace_urDeviceGetGlobalTimestampsForProfilingTag);
  mock::getCallbacks().set_after_callback("urEventWait",
                                          &after_urEventWaitForProfilingTag);
  mock::getCallbacks().set_replace_callback(
      "urEventGetProfilingInfo",
      &replace_urEventGetProfilingInfoForProfilingTag);

  sycl::property_list Props{sycl::property::queue::enable_profiling{}};
  if (IsInOrder)
    Props = {sycl::property::queue::enable_profiling{},
             sycl::property::queue::in_order{}};
  sycl::queue Queue{sycl::context{sycl::platform()}, sycl::default_selector_v,
                    Props};
  sycl::event E = sycl::ext::oneapi::experimental::submit_profiling_tag(Queue);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(E)->isProfilingTagEvent());
  ASSERT_EQ(counter_urEnqueueTimestampRecordingExp, 1u);
  LastWaitedProfilingTagEvent = nullptr;

  // Query submit first, without an explicit wait. All three values describe
  // the completion of the empty tag command, not separate barrier timestamps.
  uint64_t SubmitTime = 0;
  ASSERT_NO_THROW(
      SubmitTime =
          E.get_profiling_info<sycl::info::event_profiling::command_submit>());
  EXPECT_EQ(SubmitTime, ProfilingTagEndTime);
  EXPECT_EQ(LastWaitedProfilingTagEvent,
            sycl::detail::getSyclObjImpl(E)->getHandle());
  EXPECT_EQ(E.get_profiling_info<sycl::info::event_profiling::command_start>(),
            ProfilingTagEndTime);
  EXPECT_EQ(E.get_profiling_info<sycl::info::event_profiling::command_end>(),
            ProfilingTagEndTime);
  EXPECT_EQ(ProfilingTagGlobalTimestampQueries, 0u);
}

INSTANTIATE_TEST_SUITE_P(
    RecordingAndQueueOrder, ProfilingTagTimestampTest,
    ::testing::Combine(::testing::Bool(), ::testing::Bool()),
    [](const ::testing::TestParamInfo<ProfilingTagTimestampTest::ParamType>
           &Info) {
      return std::string(std::get<0>(Info.param) ? "Native" : "Fallback") +
             (std::get<1>(Info.param) ? "InOrder" : "OutOfOrder");
    });

TEST_F(ProfilingTagTest, RegularEventPreservesProfilingTimestamps) {
  mock::getCallbacks().set_replace_callback(
      "urDeviceGetGlobalTimestamps",
      &replace_urDeviceGetGlobalTimestampsForProfilingTag);
  mock::getCallbacks().set_after_callback("urEventWait",
                                          &after_urEventWaitForProfilingTag);
  mock::getCallbacks().set_replace_callback(
      "urEventGetProfilingInfo",
      &replace_urEventGetProfilingInfoForProfilingTag);
  sycl::queue Queue{sycl::context{sycl::platform()},
                    sycl::default_selector_v,
                    {sycl::property::queue::enable_profiling{},
                     sycl::property::queue::in_order{}}};
  sycl::event E = Queue.ext_oneapi_submit_barrier();
  ASSERT_FALSE(sycl::detail::getSyclObjImpl(E)->isProfilingTagEvent());
  LastWaitedProfilingTagEvent = nullptr;

  EXPECT_EQ(E.get_profiling_info<sycl::info::event_profiling::command_submit>(),
            ProfilingTagSubmitTime);
  EXPECT_EQ(LastWaitedProfilingTagEvent, nullptr);
  EXPECT_FALSE(LatestProfilingQuery.has_value());
  EXPECT_EQ(E.get_profiling_info<sycl::info::event_profiling::command_start>(),
            ProfilingTagStartTime);
  EXPECT_EQ(E.get_profiling_info<sycl::info::event_profiling::command_end>(),
            ProfilingTagEndTime);
  EXPECT_EQ(ProfilingTagGlobalTimestampQueries, 1u);
}

TEST_F(ProfilingTagTest, ProfilingTagTimestampFailureReleasesMarker) {
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &after_urDeviceGetInfo<true>);
  mock::getCallbacks().set_replace_callback(
      "urEnqueueTimestampRecordingExp",
      &replace_urEnqueueTimestampRecordingExpFailure);
  mock::getCallbacks().set_after_callback(
      "urEnqueueEventsWaitWithBarrier",
      &after_urEnqueueEventsWaitWithBarrierExt);
  mock::getCallbacks().set_before_callback("urEventRelease",
                                           &before_urEventRelease);

  sycl::context Ctx{sycl::platform()};
  sycl::queue Queue{Ctx, sycl::default_selector_v};

  EXPECT_THROW(sycl::ext::oneapi::experimental::submit_profiling_tag(Queue),
               sycl::exception);
  ASSERT_EQ(size_t{1}, counter_urEnqueueTimestampRecordingExp);
  ASSERT_EQ(size_t{1}, counter_urEnqueueEventsWaitWithBarrierExt);
  ASSERT_NE(nullptr, LatestBarrierEvent);
  ASSERT_TRUE(LatestBarrierEventReleased);
}
