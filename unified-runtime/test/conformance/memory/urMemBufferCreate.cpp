// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/fixtures.h"
#include "uur/raii.h"

using urMemBufferCreateTestWithFlagsParam =
    uur::urContextTestWithParam<ur_mem_flag_t>;

using urMemBufferCreateWithFlagsTest = urMemBufferCreateTestWithFlagsParam;
UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urMemBufferCreateWithFlagsTest,
    ::testing::Values(UR_MEM_FLAG_READ_WRITE, UR_MEM_FLAG_WRITE_ONLY,
                      UR_MEM_FLAG_READ_ONLY, UR_MEM_FLAG_ALLOC_HOST_POINTER),
    uur::deviceTestWithParamPrinter<ur_mem_flag_t>);

TEST_P(urMemBufferCreateWithFlagsTest, Success) {
  uur::raii::Mem buffer = nullptr;
  ASSERT_SUCCESS(
      urMemBufferCreate(context, getParam(), 4096, nullptr, buffer.ptr()));
  ASSERT_NE(nullptr, buffer);
}

TEST_P(urMemBufferCreateWithFlagsTest, InvalidNullHandleContext) {
  uur::raii::Mem buffer = nullptr;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_HANDLE,
      urMemBufferCreate(nullptr, getParam(), 4096, nullptr, buffer.ptr()));
}

TEST_P(urMemBufferCreateWithFlagsTest, InvalidNullPointerBuffer) {
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_POINTER,
      urMemBufferCreate(context, getParam(), 4096, nullptr, nullptr));
}

TEST_P(urMemBufferCreateWithFlagsTest, InvalidBufferSizeZero) {
  uur::raii::Mem buffer = nullptr;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_BUFFER_SIZE,
      urMemBufferCreate(context, getParam(), 0, nullptr, buffer.ptr()));
}

using urMemBufferCreateTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urMemBufferCreateTest);

TEST_P(urMemBufferCreateTest, InvalidEnumerationFlags) {
  uur::raii::Mem buffer = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                   urMemBufferCreate(context, UR_MEM_FLAG_FORCE_UINT32, 4096,
                                     nullptr, buffer.ptr()));
}

TEST_P(urMemBufferCreateTest, InvalidHostPtrNullProperties) {
  uur::raii::Mem buffer = nullptr;
  ur_mem_flags_t flags =
      UR_MEM_FLAG_USE_HOST_POINTER | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_HOST_PTR,
      urMemBufferCreate(context, flags, 4096, nullptr, buffer.ptr()));
}

TEST_P(urMemBufferCreateTest, InvalidHostPtrNullHost) {
  uur::raii::Mem buffer = nullptr;
  ur_mem_flags_t flags =
      UR_MEM_FLAG_USE_HOST_POINTER | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER;
  ur_buffer_properties_t properties;
  properties.pHost = nullptr;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_HOST_PTR,
      urMemBufferCreate(context, flags, 4096, &properties, buffer.ptr()));
}

TEST_P(urMemBufferCreateTest, InvalidHostPtrValidHost) {
  uur::raii::Mem buffer = nullptr;
  ur_mem_flags_t flags = 0;
  ur_buffer_properties_t properties;
  int data = 42;
  properties.pHost = &data;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_HOST_PTR,
      urMemBufferCreate(context, flags, 4096, &properties, buffer.ptr()));
}

TEST_P(urMemBufferCreateTest, CopyHostPointer) {
  std::vector<unsigned char> dataWrite{};
  dataWrite.resize(4096);
  std::vector<unsigned char> dataRead{};
  dataRead.resize(dataWrite.size());
  for (size_t i = 0; i < dataWrite.size(); i++) {
    dataWrite[i] = i & 0xff;
    dataRead[i] = 1;
  }

  ur_buffer_properties_t properties{UR_STRUCTURE_TYPE_BUFFER_PROPERTIES,
                                    nullptr, dataWrite.data()};

  ur_queue_handle_t queue;
  ASSERT_SUCCESS(urQueueCreate(context, device, nullptr, &queue));

  uur::raii::Mem buffer = nullptr;
  ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER,
                                   dataWrite.size(), &properties,
                                   buffer.ptr()));

  for (size_t i = 0; i < dataWrite.size(); i++) {
    dataWrite[i] = 2;
  }
  dataWrite.resize(0);

  ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, buffer, true, 0, dataRead.size(),
                                        dataRead.data(), 0, nullptr, nullptr));

  for (size_t i = 0; i < dataWrite.size(); i++) {
    ASSERT_EQ(dataRead[i], i & 0xff);
  }
}

TEST_P(urMemBufferCreateTest, UseHostPointer) {
  // These all copy memory instead of mapping it
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{}, uur::HIP{},
                       uur::CUDA{}, uur::OpenCL{"Intel(R) UHD Graphics 770"});

  std::vector<unsigned char> dataWrite{};
  dataWrite.resize(4096);
  std::vector<unsigned char> dataRead{};
  dataRead.resize(dataWrite.size());
  for (size_t i = 0; i < dataWrite.size(); i++) {
    dataWrite[i] = i & 0xff;
    dataRead[i] = 1;
  }

  ur_buffer_properties_t properties{UR_STRUCTURE_TYPE_BUFFER_PROPERTIES,
                                    nullptr, dataWrite.data()};

  ur_queue_handle_t queue;
  ASSERT_SUCCESS(urQueueCreate(context, device, nullptr, &queue));

  uur::raii::Mem buffer = nullptr;
  ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_USE_HOST_POINTER,
                                   dataWrite.size(), &properties,
                                   buffer.ptr()));

  for (size_t i = 0; i < dataWrite.size(); i++) {
    dataWrite[i] = i & 0x0f;
  }

  ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, buffer, true, 0, dataRead.size(),
                                        dataRead.data(), 0, nullptr, nullptr));

  for (size_t i = 0; i < dataWrite.size(); i++) {
    ASSERT_EQ(dataRead[i], i & 0x0f);
  }
}
