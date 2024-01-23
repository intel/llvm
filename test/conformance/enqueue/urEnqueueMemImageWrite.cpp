// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urEnqueueMemImageWriteTest = uur::urMemImageQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueMemImageWriteTest);

TEST_P(urEnqueueMemImageWriteTest, Success1D) {
    std::vector<uint32_t> input(width * 4, 42);
    ASSERT_SUCCESS(urEnqueueMemImageWrite(queue, image1D, true, origin,
                                          region1D, 0, 0, input.data(), 0,
                                          nullptr, nullptr));
}

TEST_P(urEnqueueMemImageWriteTest, Success2D) {
    std::vector<uint32_t> input(width * height * 4, 42);
    ASSERT_SUCCESS(urEnqueueMemImageWrite(queue, image2D, true, origin,
                                          region2D, 0, 0, input.data(), 0,
                                          nullptr, nullptr));
}

TEST_P(urEnqueueMemImageWriteTest, Success3D) {
    std::vector<uint32_t> input(width * height * depth * 4, 42);
    ASSERT_SUCCESS(urEnqueueMemImageWrite(queue, image3D, true, origin,
                                          region3D, 0, 0, input.data(), 0,
                                          nullptr, nullptr));
}

TEST_P(urEnqueueMemImageWriteTest, InvalidNullHandleQueue) {
    std::vector<uint32_t> input(width * 4, 42);
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueMemImageWrite(nullptr, image1D, true, origin,
                                            region1D, 0, 0, input.data(), 0,
                                            nullptr, nullptr));
}

TEST_P(urEnqueueMemImageWriteTest, InvalidNullHandleImage) {
    std::vector<uint32_t> input(width * 4, 42);
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueMemImageWrite(queue, nullptr, true, origin,
                                            region1D, 0, 0, input.data(), 0,
                                            nullptr, nullptr));
}

TEST_P(urEnqueueMemImageWriteTest, InvalidNullPointerSrc) {
    std::vector<uint32_t> input(width * 4, 42);
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urEnqueueMemImageWrite(queue, image1D, true, origin,
                                            region1D, 0, 0, nullptr, 0, nullptr,
                                            nullptr));
}

TEST_P(urEnqueueMemImageWriteTest, InvalidNullPtrEventWaitList) {
    std::vector<uint32_t> input(width * 4, 42);
    ASSERT_EQ_RESULT(urEnqueueMemImageWrite(queue, image1D, true, origin,
                                            region1D, 0, 0, input.data(), 1,
                                            nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t validEvent;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

    ASSERT_EQ_RESULT(urEnqueueMemImageWrite(queue, image1D, true, origin,
                                            region1D, 0, 0, input.data(), 0,
                                            &validEvent, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t inv_evt = nullptr;
    ASSERT_EQ_RESULT(urEnqueueMemImageWrite(queue, image1D, true, origin,
                                            region1D, 0, 0, input.data(), 1,
                                            &inv_evt, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ASSERT_SUCCESS(urEventRelease(validEvent));
}

TEST_P(urEnqueueMemImageWriteTest, InvalidOrigin1D) {
    std::vector<uint32_t> input(width * 4, 42);
    ur_rect_offset_t bad_origin{1, 0, 0};
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urEnqueueMemImageWrite(queue, image1D, true, bad_origin,
                                            region1D, 0, 0, input.data(), 0,
                                            nullptr, nullptr));
}

TEST_P(urEnqueueMemImageWriteTest, InvalidOrigin2D) {
    std::vector<uint32_t> input(width * height * 4, 42);
    ur_rect_offset_t bad_origin{0, 1, 0};
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urEnqueueMemImageWrite(queue, image2D, true, bad_origin,
                                            region2D, 0, 0, input.data(), 0,
                                            nullptr, nullptr));
}

TEST_P(urEnqueueMemImageWriteTest, InvalidOrigin3D) {
    std::vector<uint32_t> input(width * height * depth * 4, 42);
    ur_rect_offset_t bad_origin{0, 0, 1};
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urEnqueueMemImageWrite(queue, image3D, true, bad_origin,
                                            region3D, 0, 0, input.data(), 0,
                                            nullptr, nullptr));
}

TEST_P(urEnqueueMemImageWriteTest, InvalidRegion1D) {
    std::vector<uint32_t> input(width * 4, 42);
    ur_rect_region_t bad_region{width + 1, 1, 1};
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urEnqueueMemImageWrite(queue, image1D, true, origin,
                                            bad_region, 0, 0, input.data(), 0,
                                            nullptr, nullptr));
}

TEST_P(urEnqueueMemImageWriteTest, InvalidRegion2D) {
    std::vector<uint32_t> input(width * height * 4, 42);
    ur_rect_region_t bad_region{width, height + 1, 1};
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urEnqueueMemImageWrite(queue, image2D, true, origin,
                                            bad_region, 0, 0, input.data(), 0,
                                            nullptr, nullptr));
}

TEST_P(urEnqueueMemImageWriteTest, InvalidRegion3D) {
    std::vector<uint32_t> input(width * height * depth * 4, 42);
    ur_rect_region_t bad_region{width, height, depth + 1};
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urEnqueueMemImageWrite(queue, image3D, true, origin,
                                            bad_region, 0, 0, input.data(), 0,
                                            nullptr, nullptr));
}
