// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "../enqueue/helpers.h"
#include "fixtures.h"
#include <numeric>
#include <uur/known_failure.h>

// Choose parameters so that we get good coverage and catch some edge cases.
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
  // Tests that a 16x16x1 region can be read from a 16x16x1 device buffer at
  // offset {0,0,0} to a 16x16x1 host buffer at offset {0,0,0}.
  PARAMETERIZATION(write_whole_buffer_2D, 256, 256, (ur_rect_offset_t{0, 0, 0}),
                   (ur_rect_offset_t{0, 0, 0}), (ur_rect_region_t{16, 16, 1}),
                   16, 256, 16, 256);
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
                   (ur_rect_offset_t{0, 0, 0}), (ur_rect_region_t{8, 8, 8}), 8,
                   64, 8, 64);
  // Tests that a 4x3x2 region can be read from a 8x8x8 device buffer at
  // offset {1,2,3} to a 8x8x8 host buffer at offset {4,1,3}.
  PARAMETERIZATION(write_3d_with_offsets, 512, 512, (ur_rect_offset_t{1, 2, 3}),
                   (ur_rect_offset_t{4, 1, 3}), (ur_rect_region_t{4, 3, 2}), 8,
                   64, 8, 64);
  // Tests that a 4x16x2 region can be read from a 8x32x1 device buffer at
  // offset {1,2,0} to a 8x32x4 host buffer at offset {4,1,3}.
  // PARAMETERIZATION(write_2d_3d, 256, 1024, (ur_rect_offset_t{1, 2, 0}),
  //                  (ur_rect_offset_t{4, 1, 3}), (ur_rect_region_t{4, 16, 1}), 8,
  //                  256, 8, 256);
  // Tests that a 1x4x1 region can be read from a 8x16x4 device buffer at
  // offset {7,3,3} to a 2x8x1 host buffer at offset {1,3,0}.
  // PARAMETERIZATION(write_3d_2d, 512, 16, (ur_rect_offset_t{7, 3, 3}),
  //                  (ur_rect_offset_t{1, 3, 0}), (ur_rect_region_t{1, 4, 1}), 8,
  //                  128, 2, 16);
#undef PARAMETERIZATION
  return parameterizations;
}

struct urCommandBufferAppendMemBufferReadRectTestWithParam
    : public uur::command_buffer::urCommandBufferExpTestWithParam<
          uur::test_parameters_t> {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::urCommandBufferExpTestWithParam<
            uur::test_parameters_t>::SetUp());

    host_size = getParam().src_size;
    buffer_size = getParam().dst_size;
    host_origin = getParam().src_origin;
    buffer_origin = getParam().dst_origin;
    region = getParam().region;
    host_row_pitch = getParam().src_row_pitch;
    host_slice_pitch = getParam().src_slice_pitch;
    buffer_row_pitch = getParam().dst_row_pitch;
    buffer_slice_pitch = getParam().dst_slice_pitch;
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE,
                                     buffer_size, nullptr, &buffer));

    ASSERT_NE(buffer, nullptr);
  }

  void TearDown() override {
    if (buffer) {
      EXPECT_SUCCESS(urMemRelease(buffer));
    }

    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::urCommandBufferExpTestWithParam<
            uur::test_parameters_t>::TearDown());
  }
  size_t host_size = 0, buffer_size = 0;
  ur_rect_offset_t host_origin, buffer_origin;
  ur_rect_region_t region;
  size_t host_row_pitch, host_slice_pitch, buffer_row_pitch, buffer_slice_pitch;
  ur_mem_handle_t buffer = nullptr;
};

UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urCommandBufferAppendMemBufferReadRectTestWithParam,
    testing::ValuesIn(generateParameterizations()),
    uur::printRectTestString<
        urCommandBufferAppendMemBufferReadRectTestWithParam>);

TEST_P(urCommandBufferAppendMemBufferReadRectTestWithParam, Success) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

  // The input will just be sequentially increasing values.
  std::vector<uint8_t> input(buffer_size, 0x0);
  std::iota(std::begin(input), std::end(input), 0x0);
  EXPECT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, /* isBlocking */ true,
                                         0, input.size(), input.data(), 0,
                                         nullptr, nullptr));

  // Enqueue the rectangular read.
  std::vector<uint8_t> output(host_size, 0x0);
  EXPECT_SUCCESS(urCommandBufferAppendMemBufferReadRectExp(
      cmd_buf_handle, buffer, buffer_origin, host_origin, region,
      buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch,
      output.data(), 0, nullptr, 0, nullptr, nullptr, nullptr, nullptr));
  ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));

  ASSERT_SUCCESS(
      urEnqueueCommandBufferExp(queue, cmd_buf_handle, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));
  // Do host side equivalent.
  std::vector<uint8_t> expected(host_size, 0x0);
  uur::copyRect(input, buffer_origin, host_origin, region, buffer_row_pitch,
                buffer_slice_pitch, host_row_pitch, host_slice_pitch, expected);

  // Verify the results.
  EXPECT_EQ(expected, output);
}
