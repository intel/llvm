// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "helpers.h"
#include <numeric>
#include <uur/known_failure.h>

static std::vector<uur::test_parameters_t> generateParameterizations() {
  std::vector<uur::test_parameters_t> parameterizations;

// Choose parameters so that we get good coverage and catch some edge cases.
#define PARAMETERIZATION(name, src_buffer_size, dst_buffer_size, src_origin,   \
                         dst_origin, region, src_row_pitch, src_slice_pitch,   \
                         dst_row_pitch, dst_slice_pitch)                       \
  uur::test_parameters_t                                                       \
      name{#name,         src_buffer_size, dst_buffer_size, src_origin,        \
           dst_origin,    region,          src_row_pitch,   src_slice_pitch,   \
           dst_row_pitch, dst_slice_pitch};                                    \
  parameterizations.push_back(name);                                           \
  (void)0
  // Tests that a 16x16x1 region can be written from a 16x16x1 host buffer at
  // offset {0,0,0} to a 16x16x1 device buffer at offset {0,0,0}.
  PARAMETERIZATION(write_whole_buffer_2D, 256, 256, (ur_rect_offset_t{0, 0, 0}),
                   (ur_rect_offset_t{0, 0, 0}), (ur_rect_region_t{16, 16, 1}),
                   16, 256, 16, 256);
  // Tests that a 2x2x1 region can be written from a 4x4x1 host buffer at
  // offset {2,2,0} to a 8x4x1 device buffer at offset {4,2,0}.
  PARAMETERIZATION(write_non_zero_offsets_2D, 16, 32,
                   (ur_rect_offset_t{2, 2, 0}), (ur_rect_offset_t{4, 2, 0}),
                   (ur_rect_region_t{2, 2, 1}), 4, 16, 8, 32);
  // Tests that a 4x4x1 region can be written from a 4x4x16 host buffer at
  // offset {0,0,0} to a 8x4x16 device buffer at offset {4,0,0}.
  PARAMETERIZATION(write_different_buffer_sizes_2D, 256, 512,
                   (ur_rect_offset_t{0, 0, 0}), (ur_rect_offset_t{4, 0, 0}),
                   (ur_rect_region_t{4, 4, 1}), 4, 16, 8, 32);
  // Tests that a 1x256x1 region can be written from a 1x256x1 host buffer at
  // offset {0,0,0} to a 2x256x1 device buffer at offset {1,0,0}.
  PARAMETERIZATION(write_column_2D, 256, 512, (ur_rect_offset_t{0, 0, 0}),
                   (ur_rect_offset_t{1, 0, 0}), (ur_rect_region_t{1, 256, 1}),
                   1, 256, 2, 512);
  // Tests that a 256x1x1 region can be written from a 256x1x1 host buffer at
  // offset {0,0,0} to a 256x2x1 device buffer at offset {0,1,0}.
  PARAMETERIZATION(write_row_2D, 256, 512, (ur_rect_offset_t{0, 0, 0}),
                   (ur_rect_offset_t{0, 1, 0}), (ur_rect_region_t{256, 1, 1}),
                   256, 256, 256, 512);
  // Tests that a 8x8x8 region can be written from a 8x8x8 host buffer at
  // offset {0,0,0} to a 8x8x8 device buffer at offset {0,0,0}.
  PARAMETERIZATION(write_3D, 512, 512, (ur_rect_offset_t{0, 0, 0}),
                   (ur_rect_offset_t{0, 0, 0}), (ur_rect_region_t{8, 8, 8}), 8,
                   64, 8, 64);
  // Tests that a 4x3x2 region can be written from a 8x8x8 host buffer at
  // offset {1,2,3} to a 8x8x8 device buffer at offset {4,1,3}.
  PARAMETERIZATION(write_3D_with_offsets, 512, 512, (ur_rect_offset_t{1, 2, 3}),
                   (ur_rect_offset_t{4, 1, 3}), (ur_rect_region_t{4, 3, 2}), 8,
                   64, 8, 64);
  // Tests that a 4x16x2 region can be written from a 8x32x1 host buffer at
  // offset {1,2,0} to a 8x32x4 device buffer at offset {4,1,3}.
  PARAMETERIZATION(write_2D_3D, 256, 1024, (ur_rect_offset_t{1, 2, 0}),
                   (ur_rect_offset_t{4, 1, 3}), (ur_rect_region_t{4, 16, 1}), 8,
                   256, 8, 256);
  // Tests that a 1x4x1 region can be written from a 8x16x4 host buffer at
  // offset {7,3,3} to a 2x8x1 device buffer at offset {1,3,0}.
  PARAMETERIZATION(write_3D_2D, 512, 16, (ur_rect_offset_t{7, 3, 3}),
                   (ur_rect_offset_t{1, 3, 0}), (ur_rect_region_t{1, 4, 1}), 8,
                   128, 2, 16);
#undef PARAMETERIZATION
  return parameterizations;
}

struct urEnqueueMemBufferWriteRectTestWithParam
    : public uur::urQueueTestWithParam<uur::test_parameters_t> {};

UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urEnqueueMemBufferWriteRectTestWithParam,
    testing::ValuesIn(generateParameterizations()),
    uur::printRectTestString<urEnqueueMemBufferWriteRectTestWithParam>);

TEST_P(urEnqueueMemBufferWriteRectTestWithParam, Success) {
  const auto name = getParam().name;
  if (name.find("write_row_2D") != std::string::npos) {
    UUR_KNOWN_FAILURE_ON(uur::HIP{});
  }

  if (name.find("write_3D_2D") != std::string::npos) {
    UUR_KNOWN_FAILURE_ON(uur::HIP{});
  }

  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  // Unpack the parameters.
  const auto host_size = getParam().src_size;
  const auto buffer_size = getParam().dst_size;
  const auto host_origin = getParam().src_origin;
  const auto buffer_origin = getParam().dst_origin;
  const auto region = getParam().region;
  const auto host_row_pitch = getParam().src_row_pitch;
  const auto host_slice_pitch = getParam().src_slice_pitch;
  const auto buffer_row_pitch = getParam().dst_row_pitch;
  const auto buffer_slice_pitch = getParam().dst_slice_pitch;

  // Create a buffer we will read from.
  ur_mem_handle_t buffer = nullptr;
  ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, buffer_size,
                                   nullptr, &buffer));

  // Zero it to begin with since the write may not cover the whole buffer.
  const uint8_t zero = 0x0;
  ASSERT_SUCCESS(urEnqueueMemBufferFill(queue, buffer, &zero, sizeof(zero), 0,
                                        buffer_size, 0, nullptr, nullptr));

  // Create a host buffer of sequentially increasing values.
  std::vector<uint8_t> input(host_size, 0x0);
  std::iota(std::begin(input), std::end(input), 0x0);

  // Enqueue the rectangular write from that host buffer.
  EXPECT_SUCCESS(urEnqueueMemBufferWriteRect(
      queue, buffer, /* isBlocking */ true, buffer_origin, host_origin, region,
      buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch,
      input.data(), 0, nullptr, nullptr));

  std::vector<uint8_t> output(buffer_size, 0x0);
  EXPECT_SUCCESS(urEnqueueMemBufferRead(queue, buffer, /* is_blocking */ true,
                                        0, buffer_size, output.data(), 0,
                                        nullptr, nullptr));

  // Do host side equivalent.
  std::vector<uint8_t> expected(buffer_size, 0x0);
  uur::copyRect(input, host_origin, buffer_origin, region, host_row_pitch,
                host_slice_pitch, buffer_row_pitch, buffer_slice_pitch,
                expected);

  // Verify the results.
  EXPECT_EQ(expected, output);

  // Cleanup.
  EXPECT_SUCCESS(urMemRelease(buffer));
}

struct urEnqueueMemBufferWriteRectTest : public uur::urMemBufferQueueTest {
  ur_rect_offset_t buffer_offset{0, 0, 0};
  const ur_rect_offset_t host_offset{0, 0, 0};
  ur_rect_region_t region{size, 1, 1};
  size_t buffer_row_pitch = size;
  size_t buffer_slice_pitch = size;
  const size_t host_row_pitch = size;
  size_t host_slice_pitch = size;
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEnqueueMemBufferWriteRectTest);

TEST_P(urEnqueueMemBufferWriteRectTest, InvalidNullHandleQueue) {
  std::vector<uint32_t> src(count);
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_HANDLE,
      urEnqueueMemBufferWriteRect(nullptr, buffer, true, buffer_offset,
                                  host_offset, region, size, size, size, size,
                                  src.data(), 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferWriteRectTest, InvalidNullHandleBuffer) {
  std::vector<uint32_t> src(count);
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_HANDLE,
      urEnqueueMemBufferWriteRect(queue, nullptr, true, buffer_offset,
                                  host_offset, region, size, size, size, size,
                                  src.data(), 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferWriteRectTest, InvalidNullPointerSrc) {
  std::vector<uint32_t> src(count);
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urEnqueueMemBufferWriteRect(
                       queue, buffer, true, buffer_offset, host_offset, region,
                       size, size, size, size, nullptr, 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferWriteRectTest, InvalidNullPtrEventWaitList) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  std::vector<uint32_t> src(count);
  ASSERT_EQ_RESULT(urEnqueueMemBufferWriteRect(
                       queue, buffer, true, buffer_offset, host_offset, region,
                       size, size, size, size, src.data(), 1, nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

  ur_event_handle_t validEvent;
  ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

  ASSERT_EQ_RESULT(
      urEnqueueMemBufferWriteRect(queue, buffer, true, buffer_offset,
                                  host_offset, region, size, size, size, size,
                                  src.data(), 0, &validEvent, nullptr),
      UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

  ur_event_handle_t inv_evt = nullptr;
  ASSERT_EQ_RESULT(
      urEnqueueMemBufferWriteRect(queue, buffer, true, buffer_offset,
                                  host_offset, region, size, size, size, size,
                                  src.data(), 1, &inv_evt, nullptr),
      UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

  ASSERT_SUCCESS(urEventRelease(validEvent));
}

TEST_P(urEnqueueMemBufferWriteRectTest, InvalidSize) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  std::vector<uint32_t> src(count);
  std::fill(src.begin(), src.end(), 1);

  // region.width == 0 || region.height == 0 || region.width == 0
  region.width = 0;
  ASSERT_EQ_RESULT(urEnqueueMemBufferWriteRect(
                       queue, buffer, true, buffer_offset, host_offset, region,
                       buffer_row_pitch, buffer_slice_pitch, host_row_pitch,
                       host_slice_pitch, src.data(), 0, nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);

  // bufferRowPitch != 0 && bufferRowPitch < region.width
  region.width = buffer_row_pitch + 1;
  ASSERT_EQ_RESULT(urEnqueueMemBufferWriteRect(
                       queue, buffer, true, buffer_offset, host_offset, region,
                       buffer_row_pitch, buffer_slice_pitch, host_row_pitch,
                       host_slice_pitch, src.data(), 0, nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);

  // hostRowPitch != 0 && hostRowPitch < region.width
  region.width = host_row_pitch + 1;
  ASSERT_EQ_RESULT(urEnqueueMemBufferWriteRect(
                       queue, buffer, true, buffer_offset, host_offset, region,
                       buffer_row_pitch, buffer_slice_pitch, host_row_pitch,
                       host_slice_pitch, src.data(), 0, nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);

  // bufferSlicePitch != 0 && bufferSlicePitch < region.height * (bufferRowPitch
  // != 0 ? bufferRowPitch : region.width)
  region.width = size;
  buffer_row_pitch = 16;
  buffer_slice_pitch = (region.height * buffer_row_pitch) - 1;
  ASSERT_EQ_RESULT(urEnqueueMemBufferWriteRect(
                       queue, buffer, true, buffer_offset, host_offset, region,
                       buffer_row_pitch, buffer_slice_pitch, host_row_pitch,
                       host_slice_pitch, src.data(), 0, nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);

  // bufferSlicePitch != 0 && bufferSlicePitch % (bufferRowPitch != 0 ?
  // bufferRowPitch : region.width) != 0
  buffer_slice_pitch = size + 1;
  ASSERT_EQ_RESULT(urEnqueueMemBufferWriteRect(
                       queue, buffer, true, buffer_offset, host_offset, region,
                       buffer_row_pitch, buffer_slice_pitch, host_row_pitch,
                       host_slice_pitch, src.data(), 0, nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);

  // hostSlicePitch != 0 && hostSlicePitch < region.height * (hostRowPitch != 0
  // ? hostRowPitch : region.width)
  buffer_slice_pitch = size;
  host_slice_pitch = (region.height * host_row_pitch) - 1;
  ASSERT_EQ_RESULT(urEnqueueMemBufferWriteRect(
                       queue, buffer, true, buffer_offset, host_offset, region,
                       buffer_row_pitch, buffer_slice_pitch, host_row_pitch,
                       host_slice_pitch, src.data(), 0, nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);

  // hostSlicePitch != 0 && hostSlicePitch % (hostRowPitch != 0 ? hostRowPitch :
  // region.width) != 0
  host_slice_pitch = size + 1;
  ASSERT_EQ_RESULT(urEnqueueMemBufferWriteRect(
                       queue, buffer, true, buffer_offset, host_offset, region,
                       buffer_row_pitch, buffer_slice_pitch, host_row_pitch,
                       host_slice_pitch, src.data(), 0, nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);

  // If the combination of bufferOrigin, region, bufferRowPitch, and
  // bufferSlicePitch results in an out-of-bounds access.
  buffer_offset = {std::numeric_limits<uint64_t>::max(), 1, 1};

  ASSERT_EQ_RESULT(urEnqueueMemBufferWriteRect(
                       queue, buffer, true, buffer_offset, host_offset, region,
                       size, size, size, size, src.data(), 0, nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);
}
