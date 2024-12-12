// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include "uur/known_failure.h"

/* Using urEventReferenceTest to be able to release the event during the test */
using urEventSetCallbackTest = uur::event::urEventReferenceTest;

/**
 * Checks that the callback function is called.
 */
TEST_P(urEventSetCallbackTest, Success) {
    UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{}, uur::LevelZero{},
                         uur::LevelZeroV2{}, uur::NativeCPU{});

    struct Callback {
        static void callback([[maybe_unused]] ur_event_handle_t hEvent,
                             [[maybe_unused]] ur_execution_info_t execStatus,
                             void *pUserData) {

            auto status = reinterpret_cast<bool *>(pUserData);
            *status = true;
        }
    };

    bool didRun = false;
    ASSERT_SUCCESS(urEventSetCallback(
        event, ur_execution_info_t::UR_EXECUTION_INFO_COMPLETE,
        Callback::callback, &didRun));

    ASSERT_SUCCESS(urEventWait(1, &event));
    ASSERT_SUCCESS(urEventRelease(event));
    ASSERT_TRUE(didRun);
}

/**
 * Check that the callback function parameters are correct
 */
TEST_P(urEventSetCallbackTest, ValidateParameters) {
    UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{}, uur::LevelZero{},
                         uur::LevelZeroV2{}, uur::OpenCL{}, uur::NativeCPU{});

    struct CallbackParameters {
        ur_event_handle_t event;
        ur_execution_info_t execStatus;
    };

    struct Callback {
        static void callback(ur_event_handle_t hEvent,
                             ur_execution_info_t execStatus, void *pUserData) {

            auto parameters = reinterpret_cast<CallbackParameters *>(pUserData);
            parameters->event = hEvent;
            parameters->execStatus = execStatus;
        }
    };

    CallbackParameters parameters{};

    ASSERT_SUCCESS(urEventSetCallback(
        event, ur_execution_info_t::UR_EXECUTION_INFO_COMPLETE,
        Callback::callback, &parameters));

    ASSERT_SUCCESS(urEventWait(1, &event));
    ASSERT_SUCCESS(urEventRelease(event));
    ASSERT_EQ(event, parameters.event);
    ASSERT_EQ(ur_execution_info_t::UR_EXECUTION_INFO_COMPLETE,
              parameters.execStatus);
}

/**
 * Check that the callback function is called for each execution state.
 */
TEST_P(urEventSetCallbackTest, AllStates) {
    UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{}, uur::LevelZero{},
                         uur::LevelZeroV2{}, uur::NativeCPU{});

    struct CallbackStatus {
        bool submitted = false;
        bool running = false;
        bool complete = false;
    };

    struct Callback {
        static void callback([[maybe_unused]] ur_event_handle_t hEvent,
                             ur_execution_info_t execStatus, void *pUserData) {

            auto status = reinterpret_cast<CallbackStatus *>(pUserData);
            switch (execStatus) {
            case ur_execution_info_t::UR_EXECUTION_INFO_SUBMITTED: {
                status->submitted = true;
                break;
            }
            case ur_execution_info_t::UR_EXECUTION_INFO_RUNNING: {
                status->running = true;
                break;
            }
            case ur_execution_info_t::UR_EXECUTION_INFO_COMPLETE: {
                status->complete = true;
                break;
            }
            default: {
                FAIL() << "Invalid execution info enumeration";
            }
            }
        }
    };

    CallbackStatus status{};

    ASSERT_SUCCESS(urEventSetCallback(
        event, ur_execution_info_t::UR_EXECUTION_INFO_SUBMITTED,
        Callback::callback, &status));
    ASSERT_SUCCESS(urEventSetCallback(
        event, ur_execution_info_t::UR_EXECUTION_INFO_RUNNING,
        Callback::callback, &status));
    ASSERT_SUCCESS(urEventSetCallback(
        event, ur_execution_info_t::UR_EXECUTION_INFO_COMPLETE,
        Callback::callback, &status));

    ASSERT_SUCCESS(urEventWait(1, &event));
    ASSERT_SUCCESS(urEventRelease(event));

    ASSERT_TRUE(status.submitted);
    ASSERT_TRUE(status.running);
    ASSERT_TRUE(status.complete);
}

/**
 * Check that the callback function is called even when the event has already
 * completed
 */
TEST_P(urEventSetCallbackTest, EventAlreadyCompleted) {
    UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{}, uur::LevelZero{},
                         uur::LevelZeroV2{}, uur::NativeCPU{});

    ASSERT_SUCCESS(urEventWait(1, &event));

    struct Callback {
        static void callback([[maybe_unused]] ur_event_handle_t hEvent,
                             [[maybe_unused]] ur_execution_info_t execStatus,
                             void *pUserData) {

            auto status = reinterpret_cast<bool *>(pUserData);
            *status = true;
        }
    };

    bool didRun = false;

    ASSERT_SUCCESS(urEventSetCallback(
        event, ur_execution_info_t::UR_EXECUTION_INFO_COMPLETE,
        Callback::callback, &didRun));

    ASSERT_SUCCESS(urEventRelease(event));
    ASSERT_TRUE(didRun);
}

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEventSetCallbackTest);

/* Negative tests */
using urEventSetCallbackNegativeTest = uur::event::urEventTest;

void emptyCallback(ur_event_handle_t, ur_execution_info_t, void *) {}

TEST_P(urEventSetCallbackNegativeTest, InvalidNullHandleEvent) {
    ASSERT_EQ_RESULT(urEventSetCallback(
                         nullptr, ur_execution_info_t::UR_EXECUTION_INFO_QUEUED,
                         emptyCallback, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEventSetCallbackNegativeTest, InvalidNullPointerCallback) {
    ASSERT_EQ_RESULT(
        urEventSetCallback(event, ur_execution_info_t::UR_EXECUTION_INFO_QUEUED,
                           nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urEventSetCallbackNegativeTest, InvalidEnumeration) {
    ASSERT_EQ_RESULT(
        urEventSetCallback(event,
                           ur_execution_info_t::UR_EXECUTION_INFO_FORCE_UINT32,
                           emptyCallback, nullptr),
        UR_RESULT_ERROR_INVALID_ENUMERATION);
}

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEventSetCallbackNegativeTest);
