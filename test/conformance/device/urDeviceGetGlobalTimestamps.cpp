// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <chrono>
#include <thread>
#include <type_traits>
#include <uur/fixtures.h>

// WARNING  - This is the precision that is used in the OpenCL-CTS.
//          - We might need to modify this value per-adapter.
//          - Currently we are saying the error between host/device
//          - timers is 0.5%
constexpr double allowedTimerError = 0.005;
constexpr size_t delayTimerMultiplier = 100;

/// @brief Return the absolute difference between two numeric values.
/// @tparam T An numeric type.
/// @param[in] a First value.
/// @param[in] b Second value.
/// @return The absolute difference between `a` and `b`.
template <class T,
          typename std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
T absolute_difference(T a, T b) {
    return std::max(a, b) - std::min(a, b);
}

using urDeviceGetGlobalTimestampTest = uur::urDeviceTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urDeviceGetGlobalTimestampTest);

TEST_P(urDeviceGetGlobalTimestampTest, Success) {
    uint64_t device_time = 0;
    uint64_t host_time = 0;
    ASSERT_SUCCESS(
        urDeviceGetGlobalTimestamps(device, &device_time, &host_time));
    ASSERT_NE(device_time, 0);
    ASSERT_NE(host_time, 0);
}

TEST_P(urDeviceGetGlobalTimestampTest, SuccessHostTimer) {
    uint64_t host_time = 0;
    ASSERT_SUCCESS(urDeviceGetGlobalTimestamps(device, nullptr, &host_time));
    ASSERT_NE(host_time, 0);
}

TEST_P(urDeviceGetGlobalTimestampTest, SuccessNoTimers) {
    ASSERT_SUCCESS(urDeviceGetGlobalTimestamps(device, nullptr, nullptr));
}

TEST_P(urDeviceGetGlobalTimestampTest, SuccessSynchronizedTime) {
    // get the timer resolution of the device
    size_t deviceTimerResolutionNanoSecs = 0;
    ASSERT_SUCCESS(uur::GetDeviceProfilingTimerResolution(
        device, deviceTimerResolutionNanoSecs));
    size_t delayAmountNanoSecs =
        delayTimerMultiplier * deviceTimerResolutionNanoSecs;

    uint64_t deviceStartTime = 0, deviceEndTime = 0;
    uint64_t hostStartTime = 0, hostEndTime = 0;
    uint64_t hostOnlyStartTime = 0, hostOnlyEndTime = 0;

    ASSERT_SUCCESS(
        urDeviceGetGlobalTimestamps(device, &deviceStartTime, &hostStartTime));
    ASSERT_SUCCESS(
        urDeviceGetGlobalTimestamps(device, nullptr, &hostOnlyStartTime));
    ASSERT_NE(deviceStartTime, 0);
    ASSERT_NE(hostStartTime, 0);
    ASSERT_NE(hostOnlyStartTime, 0);
    ASSERT_GE(hostOnlyStartTime, hostStartTime);

    // wait for timers to increment
    std::this_thread::sleep_for(std::chrono::nanoseconds(delayAmountNanoSecs));

    ASSERT_SUCCESS(
        urDeviceGetGlobalTimestamps(device, &deviceEndTime, &hostEndTime));
    ASSERT_SUCCESS(
        urDeviceGetGlobalTimestamps(device, nullptr, &hostOnlyEndTime));
    ASSERT_NE(deviceEndTime, 0);
    ASSERT_NE(hostEndTime, 0);
    ASSERT_NE(hostOnlyEndTime, 0);
    ASSERT_GE(hostOnlyEndTime, hostEndTime);

    // check that the timers have advanced
    ASSERT_GT(deviceEndTime, deviceStartTime);
    ASSERT_GT(hostEndTime, hostStartTime);
    ASSERT_GT(hostOnlyEndTime, hostOnlyStartTime);

    // assert that the host/devices times are synchronized to some accuracy
    const uint64_t deviceTimeDiff = deviceEndTime - deviceStartTime;
    const uint64_t hostTimeDiff = hostEndTime - hostStartTime;
    const uint64_t observedDiff =
        absolute_difference(deviceTimeDiff, hostTimeDiff);
    const uint64_t allowedDiff = static_cast<uint64_t>(
        std::min(deviceTimeDiff, hostTimeDiff) * allowedTimerError);

    ASSERT_LE(observedDiff, allowedDiff);
}

TEST_P(urDeviceGetGlobalTimestampTest, InvalidNullHandleDevice) {
    uint64_t device_time = 0;
    uint64_t host_time = 0;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urDeviceGetGlobalTimestamps(nullptr, &device_time, &host_time));
}
