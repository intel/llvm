// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "helpers.h"
#include <numeric>

// Choose parameters so that we get good coverage and catch some edge cases.
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
    PARAMETERIZATION(write_whole_buffer_2D, 256, 256,
                     (ur_rect_offset_t{0, 0, 0}), (ur_rect_offset_t{0, 0, 0}),
                     (ur_rect_region_t{16, 16, 1}), 16, 256, 16, 256);
    // Tests that a 2x2x1 region can be read from a 4x4x1 device buffer at
    // offset {2,2,0} to a 8x4x1 host buffer at offset {4,2,0}.
    PARAMETERIZATION(write_non_zero_offsets_2D, 16, 32,
                     (ur_rect_offset_t{2, 2, 0}), (ur_rect_offset_t{4, 2, 0}),
                     (ur_rect_region_t{2, 2, 1}), 4, 16, 8, 32);
    // Tests that a 4x4x1 region can be read from a 4x4x16 device buffer at
    // offset {0,0,0} to a 8x4x16 host buffer at offset {4,0,0}.
    PARAMETERIZATION(write_different_buffer_sizes_2D, 256, 512,
                     (ur_rect_offset_t{0, 0, 0}), (ur_rect_offset_t{4, 0, 0}),
                     (ur_rect_region_t{4, 4, 1}), 4, 16, 8, 32);
    // Tests that a 1x256x1 region can be read from a 1x256x1 device buffer at
    // offset {0,0,0} to a 2x256x1 host buffer at offset {1,0,0}.
    PARAMETERIZATION(write_column_2D, 256, 512, (ur_rect_offset_t{0, 0, 0}),
                     (ur_rect_offset_t{1, 0, 0}), (ur_rect_region_t{1, 256, 1}),
                     1, 256, 2, 512);
    // Tests that a 256x1x1 region can be read from a 256x1x1 device buffer at
    // offset {0,0,0} to a 256x2x1 host buffer at offset {0,1,0}.
    PARAMETERIZATION(write_row_2D, 256, 512, (ur_rect_offset_t{0, 0, 0}),
                     (ur_rect_offset_t{0, 1, 0}), (ur_rect_region_t{256, 1, 1}),
                     256, 256, 256, 512);
    // Tests that a 8x8x8 region can be read from a 8x8x8 device buffer at
    // offset {0,0,0} to a 8x8x8 host buffer at offset {0,0,0}.
    PARAMETERIZATION(write_3d, 512, 512, (ur_rect_offset_t{0, 0, 0}),
                     (ur_rect_offset_t{0, 0, 0}), (ur_rect_region_t{8, 8, 8}),
                     8, 64, 8, 64);
    // Tests that a 4x3x2 region can be read from a 8x8x8 device buffer at
    // offset {1,2,3} to a 8x8x8 host buffer at offset {4,1,3}.
    PARAMETERIZATION(write_3d_with_offsets, 512, 512,
                     (ur_rect_offset_t{1, 2, 3}), (ur_rect_offset_t{4, 1, 3}),
                     (ur_rect_region_t{4, 3, 2}), 8, 64, 8, 64);
    // Tests that a 4x16x2 region can be read from a 8x32x1 device buffer at
    // offset {1,2,0} to a 8x32x4 host buffer at offset {4,1,3}.
    PARAMETERIZATION(write_2d_3d, 256, 1024, (ur_rect_offset_t{1, 2, 0}),
                     (ur_rect_offset_t{4, 1, 3}), (ur_rect_region_t{4, 16, 1}),
                     8, 256, 8, 256);
    // Tests that a 1x4x1 region can be read from a 8x16x4 device buffer at
    // offset {7,3,3} to a 2x8x1 host buffer at offset {1,3,0}.
    PARAMETERIZATION(write_3d_2d, 512, 16, (ur_rect_offset_t{7, 3, 3}),
                     (ur_rect_offset_t{1, 3, 0}), (ur_rect_region_t{1, 4, 1}),
                     8, 128, 2, 16);
#undef PARAMETERIZATION
    return parameterizations;
}

struct urEnqueueMemBufferReadRectTestWithParam
    : public uur::urQueueTestWithParam<uur::test_parameters_t> {};

UUR_TEST_SUITE_P(
    urEnqueueMemBufferReadRectTestWithParam,
    testing::ValuesIn(generateParameterizations()),
    uur::printRectTestString<urEnqueueMemBufferReadRectTestWithParam>);

TEST_P(urEnqueueMemBufferReadRectTestWithParam, Success) {
    // Unpack the parameters.
    const auto buffer_size = getParam().src_size;
    const auto host_size = getParam().dst_size;
    const auto buffer_offset = getParam().src_origin;
    const auto host_offset = getParam().dst_origin;
    const auto region = getParam().region;
    const auto buffer_row_pitch = getParam().src_row_pitch;
    const auto buffer_slice_pitch = getParam().src_slice_pitch;
    const auto host_row_pitch = getParam().dst_row_pitch;
    const auto host_slice_pitch = getParam().dst_slice_pitch;

    // Create a buffer we will read from.
    ur_mem_handle_t buffer = nullptr;
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE,
                                     buffer_size, nullptr, &buffer));
    // The input will just be sequentially increasing values.
    std::vector<uint8_t> input(buffer_size, 0x0);
    std::iota(std::begin(input), std::end(input), 0x0);
    EXPECT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, /* isBlocking */ true,
                                           0, input.size(), input.data(), 0,
                                           nullptr, nullptr));

    // Enqueue the rectangular read.
    std::vector<uint8_t> output(host_size, 0x0);
    EXPECT_SUCCESS(urEnqueueMemBufferReadRect(
        queue, buffer, /* isBlocking */ true, buffer_offset, host_offset,
        region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch,
        host_slice_pitch, output.data(), 0, nullptr, nullptr));

    // Do host side equivalent.
    std::vector<uint8_t> expected(host_size, 0x0);
    uur::copyRect(input, buffer_offset, host_offset, region, buffer_row_pitch,
                  buffer_slice_pitch, host_row_pitch, host_slice_pitch,
                  expected);

    // Verify the results.
    EXPECT_EQ(expected, output);

    // Cleanup.
    EXPECT_SUCCESS(urMemRelease(buffer));
}

using urEnqueueMemBufferReadRectTest = uur::urMemBufferQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueMemBufferReadRectTest);
TEST_P(urEnqueueMemBufferReadRectTest, InvalidNullHandleQueue) {
    std::vector<uint32_t> dst(count);
    ur_rect_region_t region{size, 1, 1};
    ur_rect_offset_t buffer_offset{0, 0, 0};
    ur_rect_offset_t host_offset{0, 0, 0};
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urEnqueueMemBufferReadRect(nullptr, buffer, true, buffer_offset,
                                   host_offset, region, size, size, size, size,
                                   dst.data(), 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferReadRectTest, InvalidNullHandleBuffer) {
    std::vector<uint32_t> dst(count);
    ur_rect_region_t region{size, 1, 1};
    ur_rect_offset_t buffer_offset{0, 0, 0};
    ur_rect_offset_t host_offset{0, 0, 0};
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urEnqueueMemBufferReadRect(queue, nullptr, true, buffer_offset,
                                   host_offset, region, size, size, size, size,
                                   dst.data(), 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferReadRectTest, InvalidNullPointerDst) {
    std::vector<uint32_t> dst(count);
    ur_rect_region_t region{size, 1, 1};
    ur_rect_offset_t buffer_offset{0, 0, 0};
    ur_rect_offset_t host_offset{0, 0, 0};
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urEnqueueMemBufferReadRect(queue, buffer, true,
                                                buffer_offset, host_offset,
                                                region, size, size, size, size,
                                                nullptr, 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferReadRectTest, InvalidNullPtrEventWaitList) {
    std::vector<uint32_t> dst(count);
    ur_rect_region_t region{size, 1, 1};
    ur_rect_offset_t buffer_offset{0, 0, 0};
    ur_rect_offset_t host_offset{0, 0, 0};
    ASSERT_EQ_RESULT(
        urEnqueueMemBufferReadRect(queue, buffer, true, buffer_offset,
                                   host_offset, region, size, size, size, size,
                                   dst.data(), 1, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t validEvent;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

    ASSERT_EQ_RESULT(
        urEnqueueMemBufferReadRect(queue, buffer, true, buffer_offset,
                                   host_offset, region, size, size, size, size,
                                   dst.data(), 0, &validEvent, nullptr),
        UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t inv_evt = nullptr;
    ASSERT_EQ_RESULT(
        urEnqueueMemBufferReadRect(queue, buffer, true, buffer_offset,
                                   host_offset, region, size, size, size, size,
                                   dst.data(), 1, &inv_evt, nullptr),
        UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ASSERT_SUCCESS(urEventRelease(validEvent));
}

using urEnqueueMemBufferReadRectMultiDeviceTest =
    uur::urMultiDeviceMemBufferQueueTest;

TEST_F(urEnqueueMemBufferReadRectMultiDeviceTest,
       WriteRectReadDifferentQueues) {
    // First queue does a blocking write of 42 into the buffer.
    // Then a rectangular write the buffer as 1024x1x1 1D.
    std::vector<uint32_t> input(count, 42);
    ASSERT_SUCCESS(urEnqueueMemBufferWriteRect(
        queues[0], buffer, true, {0, 0, 0}, {0, 0, 0}, {size, 1, 1}, size, size,
        size, size, input.data(), 0, nullptr, nullptr));
    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queues[0], buffer, true, 0, size,
                                           input.data(), 0, nullptr, nullptr));

    // Then the remaining queues do blocking reads from the buffer. Since the
    // queues target different devices this checks that any devices memory has
    // been synchronized.
    for (unsigned i = 1; i < queues.size(); ++i) {
        const auto queue = queues[i];
        std::vector<uint32_t> output(count, 0);
        ASSERT_SUCCESS(urEnqueueMemBufferRead(
            queue, buffer, true, 0, size, output.data(), 0, nullptr, nullptr));
        ASSERT_EQ(input, output)
            << "Result on queue " << i << " did not match!";
    }
}

TEST_P(urEnqueueMemBufferReadRectTest, InvalidSize) {
    std::vector<uint32_t> dst(count);
    // out-of-bounds access with potential overflow
    ur_rect_region_t region{size, 1, 1};
    ur_rect_offset_t buffer_offset{std::numeric_limits<uint64_t>::max(), 1, 1};
    // Creating an overflow in host_offsets leads to a crash because
    // the function doesn't do bounds checking of host buffers.
    ur_rect_offset_t host_offset{0, 0, 0};

    ASSERT_EQ_RESULT(
        urEnqueueMemBufferReadRect(queue, buffer, true, buffer_offset,
                                   host_offset, region, size, size, size, size,
                                   dst.data(), 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_SIZE);
}
