// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "helpers.h"
#include <uur/fixtures.h>

using TestParametersMemcpy2D =
    std::tuple<uur::TestParameters2D, uur::USMKind, uur::USMKind>;

struct urEnqueueUSMMemcpy2DTestWithParam
    : uur::urQueueTestWithParam<TestParametersMemcpy2D> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(
            uur::urQueueTestWithParam<TestParametersMemcpy2D>::SetUp());

        const auto [in2DParams, inSrcKind, inDstKind] = getParam();
        std::tie(pitch, width, height, src_kind, dst_kind) =
            std::make_tuple(in2DParams.pitch, in2DParams.width,
                            in2DParams.height, inSrcKind, inDstKind);

        ur_device_usm_access_capability_flags_t device_usm = 0;
        ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, device_usm));
        if (!device_usm && (src_kind == uur::USMKind::Device ||
                            dst_kind == uur::USMKind::Device)) {
            GTEST_SKIP() << "Device USM is not supported";
        }

        bool memcpy2d_support = false;
        ASSERT_SUCCESS(urContextGetInfo(
            context, UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT,
            sizeof(memcpy2d_support), &memcpy2d_support, nullptr));
        if (!memcpy2d_support) {
            GTEST_SKIP() << "2D USM memcpy is not supported";
        }

        const size_t num_elements = pitch * height;
        ASSERT_SUCCESS(uur::MakeUSMAllocationByType(
            src_kind, context, device, nullptr, nullptr, num_elements, &pSrc));

        ASSERT_SUCCESS(uur::MakeUSMAllocationByType(
            dst_kind, context, device, nullptr, nullptr, num_elements, &pDst));

        ur_event_handle_t memset_event = nullptr;

        ASSERT_SUCCESS(urEnqueueUSMFill(queue, pSrc, sizeof(memset_value),
                                        &memset_value, pitch * height, 0,
                                        nullptr, &memset_event));

        ASSERT_SUCCESS(urQueueFlush(queue));
        ASSERT_SUCCESS(urEventWait(1, &memset_event));
        ASSERT_SUCCESS(urEventRelease(memset_event));
    }

    void TearDown() override {
        if (pSrc) {
            ASSERT_SUCCESS(urUSMFree(context, pSrc));
        }
        if (pDst) {
            ASSERT_SUCCESS(urUSMFree(context, pDst));
        }
        uur::urQueueTestWithParam<TestParametersMemcpy2D>::TearDown();
    }

    void verifyMemcpySucceeded() {
        std::vector<uint8_t> host_mem(pitch * height);
        const uint8_t *host_ptr = nullptr;
        if (dst_kind == uur::USMKind::Device) {
            ASSERT_SUCCESS(urEnqueueUSMMemcpy2D(queue, true, host_mem.data(),
                                                pitch, pDst, pitch, width,
                                                height, 0, nullptr, nullptr));
            host_ptr = host_mem.data();
        } else {
            host_ptr = static_cast<const uint8_t *>(pDst);
        }
        for (size_t w = 0; w < width; ++w) {
            for (size_t h = 0; h < height; ++h) {
                const size_t index = (pitch * h) + w;
                ASSERT_TRUE(*(host_ptr + index) == memset_value);
            }
        }
    }

    void *pSrc = nullptr;
    void *pDst = nullptr;
    static constexpr uint8_t memset_value = 42;
    size_t pitch = 0;
    size_t width = 0;
    size_t height = 0;
    uur::USMKind src_kind;
    uur::USMKind dst_kind;
};

static std::vector<uur::TestParameters2D> test_sizes{
    /* Everything set to 1 */
    {1, 1, 1},
    /* Height == 1 && Pitch > width */
    {1024, 256, 1},
    /* Height == 1 && Pitch == width */
    {1024, 1024, 1},
    /* Height > 1 && Pitch > width */
    {1024, 256, 256},
    /* Height > 1 && Pitch == width + 1 */
    {234, 233, 23},
    /* Height == 1 && Pitch == width + 1 */
    {234, 233, 1}};

UUR_TEST_SUITE_P(urEnqueueUSMMemcpy2DTestWithParam,
                 ::testing::Combine(::testing::ValuesIn(test_sizes),
                                    ::testing::Values(uur::USMKind::Device,
                                                      uur::USMKind::Host,
                                                      uur::USMKind::Shared),
                                    ::testing::Values(uur::USMKind::Device,
                                                      uur::USMKind::Host,
                                                      uur::USMKind::Shared)),
                 uur::print2DTestString<urEnqueueUSMMemcpy2DTestWithParam>);

TEST_P(urEnqueueUSMMemcpy2DTestWithParam, SuccessBlocking) {
    ASSERT_SUCCESS(urEnqueueUSMMemcpy2D(queue, true, pDst, pitch, pSrc, pitch,
                                        width, height, 0, nullptr, nullptr));
    ASSERT_NO_FATAL_FAILURE(verifyMemcpySucceeded());
}

TEST_P(urEnqueueUSMMemcpy2DTestWithParam, SuccessNonBlocking) {
    ur_event_handle_t memcpy_event = nullptr;
    ASSERT_SUCCESS(urEnqueueUSMMemcpy2D(queue, false, pDst, pitch, pSrc, pitch,
                                        width, height, 0, nullptr,
                                        &memcpy_event));
    ASSERT_SUCCESS(urQueueFlush(queue));
    ASSERT_SUCCESS(urEventWait(1, &memcpy_event));
    ur_event_status_t event_status;
    ASSERT_SUCCESS(uur::GetEventInfo<ur_event_status_t>(
        memcpy_event, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS, event_status));
    ASSERT_EQ(event_status, UR_EVENT_STATUS_COMPLETE);
    ASSERT_SUCCESS(urEventRelease(memcpy_event));

    ASSERT_NO_FATAL_FAILURE(verifyMemcpySucceeded());
}

using urEnqueueUSMMemcpy2DNegativeTest = urEnqueueUSMMemcpy2DTestWithParam;
UUR_TEST_SUITE_P(urEnqueueUSMMemcpy2DNegativeTest,
                 ::testing::Values(TestParametersMemcpy2D{
                     {1, 1, 1}, uur::USMKind::Device, uur::USMKind::Device}),
                 uur::print2DTestString<urEnqueueUSMMemcpy2DTestWithParam>);

TEST_P(urEnqueueUSMMemcpy2DNegativeTest, InvalidNullHandleQueue) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueUSMMemcpy2D(nullptr, true, pDst, pitch, pSrc,
                                          pitch, width, height, 0, nullptr,
                                          nullptr));
}

TEST_P(urEnqueueUSMMemcpy2DNegativeTest, InvalidNullPointer) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urEnqueueUSMMemcpy2D(queue, true, nullptr, pitch, pSrc,
                                          pitch, width, height, 0, nullptr,
                                          nullptr));
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urEnqueueUSMMemcpy2D(queue, true, pDst, pitch, nullptr,
                                          pitch, width, height, 0, nullptr,
                                          nullptr));
}

TEST_P(urEnqueueUSMMemcpy2DNegativeTest, InvalidSize) {
    // dstPitch == 0
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urEnqueueUSMMemcpy2D(queue, true, pDst, 0, pSrc, pitch,
                                          width, height, 0, nullptr, nullptr));

    // srcPitch == 0
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urEnqueueUSMMemcpy2D(queue, true, pDst, pitch, pSrc, 0,
                                          width, height, 0, nullptr, nullptr));

    // height == 0
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urEnqueueUSMMemcpy2D(queue, true, pDst, pitch, pSrc, pitch,
                                          width, 0, 0, nullptr, nullptr));

    // dstPitch < width or srcPitch < width
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urEnqueueUSMMemcpy2D(queue, true, pDst, pitch, pSrc, pitch,
                                          width + 1, height, 0, nullptr,
                                          nullptr));

    // `dstPitch * height` is higher than the allocation size of `pDst`
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urEnqueueUSMMemcpy2D(queue, true, pDst, pitch + 1, pSrc,
                                          pitch, width, height, 0, nullptr,
                                          nullptr));

    // `srcPitch * height` is higher than the allocation size of `pSrc`
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urEnqueueUSMMemcpy2D(queue, true, pDst, pitch, pSrc,
                                          pitch + 1, width, height, 0, nullptr,
                                          nullptr));
}

TEST_P(urEnqueueUSMMemcpy2DNegativeTest, InvalidEventWaitList) {
    // enqueue something to get an event
    ur_event_handle_t event = nullptr;
    uint8_t fill_pattern = 14;
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, pDst, sizeof(fill_pattern),
                                    &fill_pattern, pitch * height, 0, nullptr,
                                    &event));
    ASSERT_NE(event, nullptr);
    ASSERT_SUCCESS(urQueueFinish(queue));

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
                     urEnqueueUSMMemcpy2D(queue, true, pDst, pitch, pSrc, pitch,
                                          width, height, 1, nullptr, nullptr));
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
                     urEnqueueUSMMemcpy2D(queue, true, pDst, pitch, pSrc, pitch,
                                          width, height, 0, &event, nullptr));

    ur_event_handle_t inv_evt = nullptr;
    ASSERT_EQ_RESULT(urEnqueueUSMMemcpy2D(queue, true, pDst, pitch, pSrc, pitch,
                                          width, height, 1, &inv_evt, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ASSERT_SUCCESS(urEventRelease(event));
}
