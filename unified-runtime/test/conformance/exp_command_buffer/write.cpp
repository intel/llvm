// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

struct testParametersWrite {
  size_t size;
  size_t offset;
  size_t write_size;
};

struct urCommandBufferWriteCommandsTest
    : uur::command_buffer::urCommandBufferExpTestWithParam<
          testParametersWrite> {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::urCommandBufferExpTestWithParam<
            testParametersWrite>::SetUp());

    size = std::get<1>(GetParam()).size;
    offset = std::get<1>(GetParam()).offset;
    write_size = std::get<1>(GetParam()).write_size;
    assert(size >= offset + write_size);
    // Allocate USM pointers
    ASSERT_SUCCESS(
        urUSMDeviceAlloc(context, device, nullptr, nullptr, size, &device_ptr));
    ASSERT_NE(device_ptr, nullptr);

    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, size,
                                     nullptr, &buffer));

    ASSERT_NE(buffer, nullptr);
  }

  void TearDown() override {
    if (device_ptr) {
      EXPECT_SUCCESS(urUSMFree(context, device_ptr));
    }

    if (buffer) {
      EXPECT_SUCCESS(urMemRelease(buffer));
    }

    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::urCommandBufferExpTestWithParam<
            testParametersWrite>::TearDown());
  }

  void verifyData(const std::vector<uint8_t> &output,
                  const std::vector<uint8_t> &input) {
    for (size_t i = 0; i < write_size; ++i) {
      ASSERT_EQ(output[i + offset], input[i])
          << "Result mismatch at index: " << i;
    }
    for (size_t i = 0; i < offset; ++i) {
      ASSERT_EQ(output[i], BASE_VALUE) << "Result mismatch at index: " << i;
    }
    for (size_t i = offset + write_size; i < size; ++i) {
      ASSERT_EQ(output[i], BASE_VALUE) << "Result mismatch at index: " << i;
    }
  }

  const uint8_t BASE_VALUE = 1;
  size_t size, write_size, offset;

  void *device_ptr = nullptr;
  ur_mem_handle_t buffer = nullptr;
};

static std::vector<testParametersWrite> test_cases{
    // write whole buffer
    {1, 0, 1},
    {256, 0, 256},
    {1024, 0, 1024},
    // write part of buffer
    {256, 127, 128},
    {1024, 256, 256},
};

template <typename T>
static std::string printWriteTestString(
    const testing::TestParamInfo<typename T::ParamType> &info) {
  const auto device_handle = std::get<0>(info.param).device;
  const auto platform_device_name =
      uur::GetPlatformAndDeviceName(device_handle);
  std::stringstream test_name;
  test_name << platform_device_name << "__size__"
            << std::get<1>(info.param).size << "__offset__"
            << std::get<1>(info.param).offset << "__write_size__"
            << std::get<1>(info.param).write_size;
  return test_name.str();
}

UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urCommandBufferWriteCommandsTest, testing::ValuesIn(test_cases),
    printWriteTestString<urCommandBufferWriteCommandsTest>);

TEST_P(urCommandBufferWriteCommandsTest, Buffer) {
  std::vector<uint8_t> input(size);
  std::iota(input.begin(), input.end(), 1);

  std::vector<uint8_t> output(size, BASE_VALUE);
  ASSERT_SUCCESS(urCommandBufferAppendMemBufferWriteExp(
      cmd_buf_handle, buffer, offset, write_size, input.data(), 0, nullptr, 0,
      nullptr, nullptr, nullptr, nullptr));
  ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));

  ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                         output.data(), 0, nullptr, nullptr));
  ASSERT_SUCCESS(
      urEnqueueCommandBufferExp(queue, cmd_buf_handle, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, buffer, true, 0, size,
                                        output.data(), 0, nullptr, nullptr));
  verifyData(output, input);
}
