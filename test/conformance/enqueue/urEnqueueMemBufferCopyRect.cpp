// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "helpers.h"
#include <numeric>

static std::vector<uur::test_parameters_t> generateParameterizations() {
    std::vector<uur::test_parameters_t> parameterizations;

// Choose parameters so that we get good coverage and catch some edge cases.
#define PARAMETERIZATION(name, src_buffer_size, dst_buffer_size, src_origin,   \
                         dst_origin, region, src_row_pitch, src_slice_pitch,   \
                         dst_row_pitch, dst_slice_pitch)                       \
    uur::test_parameters_t name{                                               \
        #name,         src_buffer_size, dst_buffer_size, src_origin,           \
        dst_origin,    region,          src_row_pitch,   src_slice_pitch,      \
        dst_row_pitch, dst_slice_pitch};                                       \
    parameterizations.push_back(name);                                         \
    (void)0
    // Tests that a 16x16x1 region can be read from a 16x16x1 device buffer at
    // offset {0,0,0} to a 16x16x1 host buffer at offset {0,0,0}.
    PARAMETERIZATION(copy_whole_buffer_2D, 256, 256,
                     (ur_rect_offset_t{0, 0, 0}), (ur_rect_offset_t{0, 0, 0}),
                     (ur_rect_region_t{16, 16, 1}), 16, 256, 16, 256);
    // Tests that a 2x2x1 region can be read from a 4x4x1 device buffer at
    // offset {2,2,0} to a 8x4x1 host buffer at offset {4,2,0}.
    PARAMETERIZATION(copy_non_zero_offsets_2D, 16, 32,
                     (ur_rect_offset_t{2, 2, 0}), (ur_rect_offset_t{4, 2, 0}),
                     (ur_rect_region_t{2, 2, 1}), 4, 16, 8, 32);
    // Tests that a 4x4x1 region can be read from a 4x4x16 device buffer at
    // offset {0,0,0} to a 8x4x16 host buffer at offset {4,0,0}.
    PARAMETERIZATION(copy_different_buffer_sizes_2D, 256, 512,
                     (ur_rect_offset_t{0, 0, 0}), (ur_rect_offset_t{4, 0, 0}),
                     (ur_rect_region_t{4, 4, 1}), 4, 16, 8, 32);
    // Tests that a 1x256x1 region can be read from a 1x256x1 device buffer at
    // offset {0,0,0} to a 2x256x1 host buffer at offset {1,0,0}.
    PARAMETERIZATION(copy_column_2D, 256, 512, (ur_rect_offset_t{0, 0, 0}),
                     (ur_rect_offset_t{1, 0, 0}), (ur_rect_region_t{1, 256, 1}),
                     1, 256, 2, 512);
    // Tests that a 256x1x1 region can be read from a 256x1x1 device buffer at
    // offset {0,0,0} to a 256x2x1 host buffer at offset {0,1,0}.
    PARAMETERIZATION(copy_row_2D, 256, 512, (ur_rect_offset_t{0, 0, 0}),
                     (ur_rect_offset_t{0, 1, 0}), (ur_rect_region_t{256, 1, 1}),
                     256, 256, 256, 512);
    // Tests that a 8x8x8 region can be read from a 8x8x8 device buffer at
    // offset {0,0,0} to a 8x8x8 host buffer at offset {0,0,0}.
    PARAMETERIZATION(copy_3d, 512, 512, (ur_rect_offset_t{0, 0, 0}),
                     (ur_rect_offset_t{0, 0, 0}), (ur_rect_region_t{8, 8, 8}),
                     8, 64, 8, 64);
    // Tests that a 4x3x2 region can be read from a 8x8x8 device buffer at
    // offset {1,2,3} to a 8x8x8 host buffer at offset {4,1,3}.
    PARAMETERIZATION(copy_3d_with_offsets, 512, 512,
                     (ur_rect_offset_t{1, 2, 3}), (ur_rect_offset_t{4, 1, 3}),
                     (ur_rect_region_t{4, 3, 2}), 8, 64, 8, 64);
    // Tests that a 4x16x2 region can be read from a 8x32x1 device buffer at
    // offset {1,2,0} to a 8x32x4 host buffer at offset {4,1,3}.
    PARAMETERIZATION(copy_2d_3d, 256, 1024, (ur_rect_offset_t{1, 2, 0}),
                     (ur_rect_offset_t{4, 1, 3}), (ur_rect_region_t{4, 16, 1}),
                     8, 256, 8, 256);
    // Tests that a 1x4x1 region can be read from a 8x16x4 device buffer at
    // offset {7,3,3} to a 2x8x1 host buffer at offset {1,3,0}.
    PARAMETERIZATION(copy_3d_2d, 512, 16, (ur_rect_offset_t{7, 3, 3}),
                     (ur_rect_offset_t{1, 3, 0}), (ur_rect_region_t{1, 4, 1}),
                     8, 128, 2, 16);
#undef PARAMETERIZATION
    return parameterizations;
}

struct urEnqueueMemBufferCopyRectTestWithParam
    : public uur::urQueueTestWithParam<uur::test_parameters_t> {};

UUR_TEST_SUITE_P(
    urEnqueueMemBufferCopyRectTestWithParam,
    testing::ValuesIn(generateParameterizations()),
    uur::printRectTestString<urEnqueueMemBufferCopyRectTestWithParam>);

TEST_P(urEnqueueMemBufferCopyRectTestWithParam, Success) {
    // Unpack the parameters.
    const auto src_buffer_size = getParam().src_size;
    const auto dst_buffer_size = getParam().dst_size;
    const auto src_buffer_origin = getParam().src_origin;
    const auto dst_buffer_origin = getParam().dst_origin;
    const auto region = getParam().region;
    const auto src_buffer_row_pitch = getParam().src_row_pitch;
    const auto src_buffer_slice_pitch = getParam().src_slice_pitch;
    const auto dst_buffer_row_pitch = getParam().dst_row_pitch;
    const auto dst_buffer_slice_pitch = getParam().dst_slice_pitch;

    // Create two buffers to copy between.
    ur_mem_handle_t src_buffer = nullptr;
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE,
                                     src_buffer_size, nullptr, &src_buffer));

    // Fill src buffer with sequentially increasing values.
    std::vector<uint8_t> input(src_buffer_size, 0x0);
    std::iota(std::begin(input), std::end(input), 0x0);
    EXPECT_SUCCESS(urEnqueueMemBufferWrite(queue, src_buffer,
                                           /* is_blocking */ true, 0,
                                           src_buffer_size, input.data(), 0,
                                           nullptr, nullptr));

    ur_mem_handle_t dst_buffer = nullptr;
    EXPECT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE,
                                     dst_buffer_size, nullptr, &dst_buffer));

    // Zero destination buffer to begin with since the write may not cover the
    // whole buffer.
    const uint8_t zero = 0x0;
    EXPECT_SUCCESS(urEnqueueMemBufferFill(queue, dst_buffer, &zero,
                                          sizeof(zero), 0, dst_buffer_size, 0,
                                          nullptr, nullptr));

    // Enqueue the rectangular copy between the buffers.
    EXPECT_SUCCESS(urEnqueueMemBufferCopyRect(
        queue, src_buffer, dst_buffer, src_buffer_origin, dst_buffer_origin,
        region, src_buffer_row_pitch, src_buffer_slice_pitch,
        dst_buffer_row_pitch, dst_buffer_slice_pitch, 0, nullptr, nullptr));

    std::vector<uint8_t> output(dst_buffer_size, 0x0);
    EXPECT_SUCCESS(urEnqueueMemBufferRead(queue, dst_buffer,
                                          /* is_blocking */ true, 0,
                                          dst_buffer_size, output.data(), 0,
                                          nullptr, nullptr));

    // Do host side equivalent.
    std::vector<uint8_t> expected(dst_buffer_size, 0x0);
    uur::copyRect(input, src_buffer_origin, dst_buffer_origin, region,
                  src_buffer_row_pitch, src_buffer_slice_pitch,
                  dst_buffer_row_pitch, dst_buffer_slice_pitch, expected);

    // Verify the results.
    EXPECT_EQ(expected, output);

    // Cleanup.
    EXPECT_SUCCESS(urMemRelease(src_buffer));
    EXPECT_SUCCESS(urMemRelease(dst_buffer));
}

struct urEnqueueMemBufferCopyRectTest : uur::urQueueTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::SetUp());
        ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_WRITE_ONLY, size,
                                         nullptr, &src_buffer));
        ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_ONLY, size,
                                         nullptr, &dst_buffer));
        input.assign(count, 42);
        ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, src_buffer, true, 0, size,
                                               input.data(), 0, nullptr,
                                               nullptr));
    }

    void TearDown() override {
        if (src_buffer) {
            EXPECT_SUCCESS(urMemRelease(src_buffer));
        }
        if (src_buffer) {
            EXPECT_SUCCESS(urMemRelease(dst_buffer));
        }
        urQueueTest::TearDown();
    }

    const size_t count = 1024;
    const size_t size = sizeof(uint32_t) * count;
    ur_mem_handle_t src_buffer = nullptr;
    ur_mem_handle_t dst_buffer = nullptr;
    std::vector<uint32_t> input;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueMemBufferCopyRectTest);

TEST_P(urEnqueueMemBufferCopyRectTest, InvalidNullHandleQueue) {
    ur_rect_region_t src_region{size, 1, 1};
    ur_rect_offset_t src_origin{0, 0, 0};
    ur_rect_offset_t dst_origin{0, 0, 0};
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueMemBufferCopyRect(nullptr, src_buffer, dst_buffer,
                                                src_origin, dst_origin,
                                                src_region, size, size, size,
                                                size, 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferCopyRectTest, InvalidNullHandleBufferSrc) {
    ur_rect_region_t src_region{size, 1, 1};
    ur_rect_offset_t src_origin{0, 0, 0};
    ur_rect_offset_t dst_origin{0, 0, 0};
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueMemBufferCopyRect(queue, nullptr, dst_buffer,
                                                src_origin, dst_origin,
                                                src_region, size, size, size,
                                                size, 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferCopyRectTest, InvalidNullHandleBufferDst) {
    ur_rect_region_t src_region{size, 1, 1};
    ur_rect_offset_t src_origin{0, 0, 0};
    ur_rect_offset_t dst_origin{0, 0, 0};
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueMemBufferCopyRect(queue, src_buffer, nullptr,
                                                src_origin, dst_origin,
                                                src_region, size, size, size,
                                                size, 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferCopyRectTest, InvalidNullPtrEventWaitList) {
    ur_rect_region_t src_region{size, 1, 1};
    ur_rect_offset_t src_origin{0, 0, 0};
    ur_rect_offset_t dst_origin{0, 0, 0};
    ASSERT_EQ_RESULT(urEnqueueMemBufferCopyRect(queue, src_buffer, dst_buffer,
                                                src_origin, dst_origin,
                                                src_region, size, size, size,
                                                size, 1, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t validEvent;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

    ASSERT_EQ_RESULT(urEnqueueMemBufferCopyRect(queue, src_buffer, dst_buffer,
                                                src_origin, dst_origin,
                                                src_region, size, size, size,
                                                size, 0, &validEvent, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t inv_evt = nullptr;
    ASSERT_EQ_RESULT(urEnqueueMemBufferCopyRect(queue, src_buffer, dst_buffer,
                                                src_origin, dst_origin,
                                                src_region, size, size, size,
                                                size, 1, &inv_evt, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ASSERT_SUCCESS(urEventRelease(validEvent));
}

using urEnqueueMemBufferCopyRectMultiDeviceTest =
    uur::urMultiDeviceMemBufferQueueTest;

TEST_F(urEnqueueMemBufferCopyRectMultiDeviceTest, CopyRectReadDifferentQueues) {
    // First queue does a fill.
    const uint32_t input = 42;
    ASSERT_SUCCESS(urEnqueueMemBufferFill(queues[0], buffer, &input,
                                          sizeof(input), 0, size, 0, nullptr,
                                          nullptr));

    // Then a rectangular copy treating both buffers as 1024x1x1 1D buffers with
    // zero offsets.
    ur_mem_handle_t dst_buffer = nullptr;
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_ONLY, size,
                                     nullptr, &dst_buffer));
    EXPECT_SUCCESS(urEnqueueMemBufferCopyRect(
        queues[0], buffer, dst_buffer, {0, 0, 0}, {0, 0, 0}, {size, 1, 1}, size,
        size, size, size, 0, nullptr, nullptr));

    // Wait for the queue to finish executing.
    EXPECT_SUCCESS(urEnqueueEventsWait(queues[0], 0, nullptr, nullptr));

    // Then the remaining queues do blocking reads from the buffer. Since the
    // queues target different devices this checks that any devices memory has
    // been synchronized.
    for (unsigned i = 1; i < queues.size(); ++i) {
        const auto queue = queues[i];
        std::vector<uint32_t> output(count, 0);
        EXPECT_SUCCESS(urEnqueueMemBufferRead(queue, dst_buffer, true, 0, size,
                                              output.data(), 0, nullptr,
                                              nullptr));
        for (unsigned j = 0; j < count; ++j) {
            EXPECT_EQ(input, output[j])
                << "Result on queue " << i << " did not match at index " << j
                << "!";
        }
    }

    EXPECT_SUCCESS(urMemRelease(dst_buffer));
}

TEST_P(urEnqueueMemBufferCopyRectTest, InvalidSize) {
    // out-of-bounds access with potential overflow
    ur_rect_region_t src_region{size, 1, 1};
    ur_rect_offset_t src_origin{std::numeric_limits<uint64_t>::max(), 1, 1};
    ur_rect_offset_t dst_origin{0, 0, 0};

    ASSERT_EQ_RESULT(urEnqueueMemBufferCopyRect(queue, src_buffer, dst_buffer,
                                                src_origin, dst_origin,
                                                src_region, size, size, size,
                                                size, 0, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}
