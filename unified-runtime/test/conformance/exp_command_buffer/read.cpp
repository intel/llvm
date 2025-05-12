// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

struct testParametersRead {
  size_t size;
  size_t offset;
  size_t read_size;
};

struct urCommandBufferReadCommandsTest
    : uur::command_buffer::urCommandBufferExpTestWithParam<testParametersRead> {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::urCommandBufferExpTestWithParam<
            testParametersRead>::SetUp());

    size = std::get<1>(GetParam()).size;
    offset = std::get<1>(GetParam()).offset;
    read_size = std::get<1>(GetParam()).read_size;
    assert(size >= offset + read_size);
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
            testParametersRead>::TearDown());
  }

  void verifyData(const std::vector<uint8_t> &output,
                  const std::vector<uint8_t> &input) {
    for (size_t i = 0; i < read_size; ++i) {
      ASSERT_EQ(output[i], input[i + offset])
          << "Result mismatch at index: " << i;
    }
  }

  size_t size, read_size, offset;

  void *device_ptr = nullptr;
  ur_mem_handle_t buffer = nullptr;
};

static std::vector<testParametersRead> test_cases{
    // read whole buffer
    {1, 0, 1},
    {256, 0, 256},
    {1024, 0, 1024},
    // read part of buffer
    {256, 127, 128},
    {1024, 256, 256},
};

template <typename T>
static std::string
printReadTestString(const testing::TestParamInfo<typename T::ParamType> &info) {
  const auto device_handle = std::get<0>(info.param).device;
  const auto platform_device_name =
      uur::GetPlatformAndDeviceName(device_handle);
  std::stringstream test_name;
  test_name << platform_device_name << "__size__"
            << std::get<1>(info.param).size << "__offset__"
            << std::get<1>(info.param).offset << "__read_size__"
            << std::get<1>(info.param).read_size;
  return test_name.str();
}

UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urCommandBufferReadCommandsTest, testing::ValuesIn(test_cases),
    printReadTestString<urCommandBufferReadCommandsTest>);

TEST_P(urCommandBufferReadCommandsTest, Buffer) {
  std::vector<uint8_t> input(size);
  std::iota(input.begin(), input.end(), 1);

  std::vector<uint8_t> output(size, 1);
  ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                         input.data(), 0, nullptr, nullptr));

  ASSERT_SUCCESS(urCommandBufferAppendMemBufferReadExp(
      cmd_buf_handle, buffer, offset, read_size, output.data(), 0, nullptr, 0,
      nullptr, nullptr, nullptr, nullptr));
  ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));

  ASSERT_SUCCESS(
      urEnqueueCommandBufferExp(queue, cmd_buf_handle, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  verifyData(output, input);
}
