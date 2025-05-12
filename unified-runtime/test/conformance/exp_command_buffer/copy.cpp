// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

struct testParametersMemcpy {
  size_t size;
  size_t offset_src;
  size_t offset_dst;
  size_t copy_size;
};

struct urCommandBufferMemcpyCommandsTest
    : uur::command_buffer::urCommandBufferExpTestWithParam<
          testParametersMemcpy> {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::urCommandBufferExpTestWithParam<
            testParametersMemcpy>::SetUp());

    size = std::get<1>(GetParam()).size;
    offset_src = std::get<1>(GetParam()).offset_src;
    offset_dst = std::get<1>(GetParam()).offset_dst;
    copy_size = std::get<1>(GetParam()).copy_size;
    assert(size >= offset_src + copy_size);
    assert(size >= offset_dst + copy_size);
    // Allocate USM pointers
    ASSERT_SUCCESS(
        urUSMDeviceAlloc(context, device, nullptr, nullptr, size, &device_ptr));
    ASSERT_NE(device_ptr, nullptr);

    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, size,
                                     nullptr, &buffer));

    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, size,
                                     nullptr, &buffer_base));

    ASSERT_NE(buffer, nullptr);
    ASSERT_NE(buffer_base, nullptr);
  }

  void TearDown() override {
    if (device_ptr) {
      EXPECT_SUCCESS(urUSMFree(context, device_ptr));
    }

    if (buffer) {
      EXPECT_SUCCESS(urMemRelease(buffer));
    }

    if (buffer_base) {
      EXPECT_SUCCESS(urMemRelease(buffer_base));
    }

    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::urCommandBufferExpTestWithParam<
            testParametersMemcpy>::TearDown());
  }

  void verifyData(const std::vector<uint8_t> &output,
                  const std::vector<uint8_t> &input) {
    for (size_t i = 0; i < copy_size; ++i) {
      ASSERT_EQ(output[i + offset_dst], input[i + offset_src])
          << "Result mismatch at index: " << i;
    }
    for (size_t i = 0; i < offset_dst; ++i) {
      ASSERT_EQ(output[i], BASE_VALUE) << "Result mismatch at index: " << i;
    }
    for (size_t i = offset_dst + copy_size; i < size; ++i) {
      ASSERT_EQ(output[i], BASE_VALUE) << "Result mismatch at index: " << i;
    }
  }
  const uint8_t BASE_VALUE = 1;
  size_t size, copy_size, offset_src, offset_dst;

  ur_exp_command_buffer_sync_point_t sync_point, sync_point2;
  void *device_ptr = nullptr;
  ur_mem_handle_t buffer = nullptr;
  ur_mem_handle_t buffer_base = nullptr;
};

static std::vector<testParametersMemcpy> test_cases{
    // copy whole buffer
    {1, 0, 0, 1},
    {256, 0, 0, 256},
    {1024, 0, 0, 1024},
    // copy part of buffer
    {256, 127, 127, 128},
    {1024, 256, 256, 256},
    // copy to different offset
    {256, 127, 196, 25},
    {1024, 756, 256, 256},

};

template <typename T>
static std::string printMemcpyTestString(
    const testing::TestParamInfo<typename T::ParamType> &info) {
  const auto device_handle = std::get<0>(info.param).device;
  const auto platform_device_name =
      uur::GetPlatformAndDeviceName(device_handle);
  std::stringstream test_name;
  test_name << platform_device_name << "__size__"
            << std::get<1>(info.param).size << "__offset_src__"
            << std::get<1>(info.param).offset_src << "__offset_src__"
            << std::get<1>(info.param).offset_dst << "__copy_size__"
            << std::get<1>(info.param).copy_size;
  return test_name.str();
}

UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urCommandBufferMemcpyCommandsTest, testing::ValuesIn(test_cases),
    printMemcpyTestString<urCommandBufferMemcpyCommandsTest>);

TEST_P(urCommandBufferMemcpyCommandsTest, Buffer) {
  std::vector<uint8_t> input(size);
  std::iota(input.begin(), input.end(), 1);

  ASSERT_SUCCESS(urCommandBufferAppendMemBufferWriteExp(
      cmd_buf_handle, buffer_base, 0, size, input.data(), 0, nullptr, 0,
      nullptr, &sync_point, nullptr, nullptr));

  ASSERT_SUCCESS(urCommandBufferAppendMemBufferCopyExp(
      cmd_buf_handle, buffer_base, buffer, offset_src, offset_dst, copy_size, 1,
      &sync_point, 0, nullptr, &sync_point2, nullptr, nullptr));
  std::vector<uint8_t> output(size, BASE_VALUE);
  ASSERT_SUCCESS(urCommandBufferAppendMemBufferReadExp(
      cmd_buf_handle, buffer, 0, size, output.data(), 1, &sync_point2, 0,
      nullptr, nullptr, nullptr, nullptr));
  ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));

  ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                         output.data(), 0, nullptr, nullptr));

  ASSERT_SUCCESS(
      urEnqueueCommandBufferExp(queue, cmd_buf_handle, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  verifyData(output, input);
}

TEST_P(urCommandBufferMemcpyCommandsTest, USM) {
  std::vector<uint8_t> input(size);
  std::iota(input.begin(), input.end(), 1);
  ASSERT_SUCCESS(urCommandBufferAppendUSMMemcpyExp(
      cmd_buf_handle, ((uint8_t *)device_ptr) + offset_dst,
      input.data() + offset_src, copy_size, 0, nullptr, 0, nullptr, &sync_point,
      nullptr, nullptr));

  std::vector<uint8_t> output(size, BASE_VALUE);
  ASSERT_SUCCESS(urCommandBufferAppendUSMMemcpyExp(
      cmd_buf_handle, output.data(), device_ptr, size, 1, &sync_point, 0,
      nullptr, nullptr, nullptr, nullptr));

  ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));

  ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, true, device_ptr, output.data(),
                                    size, 0, nullptr, nullptr));

  ASSERT_SUCCESS(
      urEnqueueCommandBufferExp(queue, cmd_buf_handle, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  verifyData(output, input);
}
