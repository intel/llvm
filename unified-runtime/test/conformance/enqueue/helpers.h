// Copyright (C) 2023-2026 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UUR_ENQUEUE_RECT_HELPERS_H_INCLUDED
#define UUR_ENQUEUE_RECT_HELPERS_H_INCLUDED

#include "ur_api.h"
#include <cstring>
#include <sstream>
#include <uur/fixtures.h>

namespace uur {

struct test_parameters_t {
  std::string name;
  size_t src_size;
  size_t dst_size;
  ur_rect_offset_t src_origin;
  ur_rect_offset_t dst_origin;
  ur_rect_region_t region;
  size_t src_row_pitch;
  size_t src_slice_pitch;
  size_t dst_row_pitch;
  size_t dst_slice_pitch;
};

template <typename T>
inline std::string
printRectTestString(const testing::TestParamInfo<typename T::ParamType> &info) {
  // ParamType will be std::tuple<ur_device_handle_t, test_parameters_t>
  const auto device_handle = std::get<0>(info.param).device;
  const auto platform_device_name = GetPlatformAndDeviceName(device_handle);

  std::stringstream ss;
  auto param_tuple = std::get<1>(info.param);
  auto test_name = std::get<0>(param_tuple).name;
  auto queue_mode = std::get<1>(param_tuple);

  ss << platform_device_name << "__" << test_name << "__" << queue_mode;

  return ss.str();
}

template <typename T>
inline std::string printRectTestStringMultiQueue(
    const testing::TestParamInfo<typename T::ParamType> &info) {
  // ParamType will be std::tuple<ur_device_handle_t, test_parameters_t>
  const auto device_handle = std::get<0>(info.param).device;
  const auto platform_device_name = GetPlatformAndDeviceName(device_handle);
  auto paramTuple = std::get<1>(info.param);
  auto param = std::get<0>(paramTuple);

  auto queueMode = std::get<1>(paramTuple);

  std::stringstream test_name;
  test_name << platform_device_name;
  test_name << param.name << "__" << queueMode;

  return test_name.str();
}

// Performs host side equivalent of urEnqueueMemBufferReadRect,
// urEnqueueMemBufferWriteRect and urEnqueueMemBufferCopyRect.
inline void copyRect(std::vector<uint8_t> src, ur_rect_offset_t src_offset,
                     ur_rect_offset_t dst_offset, ur_rect_region_t region,
                     size_t src_row_pitch, size_t src_slice_pitch,
                     size_t dst_row_pitch, size_t dst_slice_pitch,
                     std::vector<uint8_t> &dst) {
  const auto src_linear_offset = src_offset.x + src_offset.y * src_row_pitch +
                                 src_offset.z * src_slice_pitch;
  const auto src_start = src.data() + src_linear_offset;

  const auto dst_linear_offset = dst_offset.x + dst_offset.y * dst_row_pitch +
                                 dst_offset.z * dst_slice_pitch;
  const auto dst_start = dst.data() + dst_linear_offset;

  for (unsigned k = 0; k < region.depth; ++k) {
    const auto src_slice = src_start + k * src_slice_pitch;
    const auto dst_slice = dst_start + k * dst_slice_pitch;
    for (unsigned j = 0; j < region.height; ++j) {
      auto src_row = src_slice + j * src_row_pitch;
      auto dst_row = dst_slice + j * dst_row_pitch;
      std::memcpy(dst_row, src_row, region.width);
    }
  }
}

struct TestParameters2D {
  size_t pitch;
  size_t width;
  size_t height;
};

template <typename T>
inline std::string
print2DTestString(const testing::TestParamInfo<typename T::ParamType> &info) {
  const auto device_handle = std::get<0>(info.param).device;
  const auto platform_device_name =
      uur::GetPlatformAndDeviceName(device_handle);
  std::stringstream test_name;

  auto paramTuple = std::get<1>(info.param);
  auto param = std::get<0>(paramTuple);
  auto queueMode = std::get<1>(paramTuple);
  const auto src_kind = std::get<1>(param);
  const auto dst_kind = std::get<2>(param);
  TestParameters2D testParams = std::get<0>(param);
  test_name << platform_device_name << "__pitch__" << testParams.pitch
            << "__width__" << testParams.width << "__height__"
            << testParams.height << "__src__" << src_kind << "__dst__"
            << dst_kind << "__" << queueMode;

  return test_name.str();
}

struct mem_buffer_test_parameters_t {
  size_t count;
  ur_mem_flag_t mem_flag;
};

static std::vector<mem_buffer_test_parameters_t> mem_buffer_test_parameters{
    {1024, UR_MEM_FLAG_READ_WRITE},
    {2500, UR_MEM_FLAG_READ_WRITE},
    {4096, UR_MEM_FLAG_READ_WRITE},
    {6000, UR_MEM_FLAG_READ_WRITE},
    {1024, UR_MEM_FLAG_WRITE_ONLY},
    {2500, UR_MEM_FLAG_WRITE_ONLY},
    {4096, UR_MEM_FLAG_WRITE_ONLY},
    {6000, UR_MEM_FLAG_WRITE_ONLY},
    {1024, UR_MEM_FLAG_READ_ONLY},
    {2500, UR_MEM_FLAG_READ_ONLY},
    {4096, UR_MEM_FLAG_READ_ONLY},
    {6000, UR_MEM_FLAG_READ_ONLY},
    {1024, UR_MEM_FLAG_ALLOC_HOST_POINTER},
    {2500, UR_MEM_FLAG_ALLOC_HOST_POINTER},
    {4096, UR_MEM_FLAG_ALLOC_HOST_POINTER},
    {6000, UR_MEM_FLAG_ALLOC_HOST_POINTER},
};

struct mem_buffer_map_write_test_parameters_t {
  size_t count;
  ur_mem_flag_t mem_flag;
  ur_map_flag_t map_flag;
};

template <typename T>
inline std::string printMemBufferTestString(
    const testing::TestParamInfo<typename T::ParamType> &info) {
  // ParamType will be std::tuple<ur_device_handle_t,
  // mem_buffer_test_parameters_t>
  const auto device_handle = std::get<0>(info.param).device;
  const auto platform_device_name = GetPlatformAndDeviceName(device_handle);

  auto paramTuple = std::get<1>(info.param);
  auto param = std::get<0>(paramTuple);
  auto queueMode = std::get<1>(paramTuple);

  std::stringstream ss;
  ss << param.count;
  ss << "_";
  ss << param.mem_flag;
  ss << "__";
  ss << queueMode;

  return platform_device_name + "__" + ss.str();
}

template <typename T>
inline std::string printMemBufferMapWriteTestString(
    const testing::TestParamInfo<typename T::ParamType> &info) {
  // ParamType will be std::tuple<ur_device_handle_t,
  // mem_buffer_map_write_test_parameters_t>
  const auto device_handle = std::get<0>(info.param).device;
  const auto platform_device_name = GetPlatformAndDeviceName(device_handle);

  auto paramTuple = std::get<1>(info.param);
  auto param = std::get<0>(paramTuple);
  auto queueMode = std::get<1>(paramTuple);

  std::stringstream ss;
  ss << param.map_flag << "__" << queueMode;

  return platform_device_name + "__" + ss.str();
}

template <typename T>
inline std::string printFillTestStringMultiQueueType(
    const testing::TestParamInfo<typename T::ParamType> &info) {
  const auto device_handle = std::get<0>(info.param).device;
  const auto platform_device_name =
      uur::GetPlatformAndDeviceName(device_handle);
  auto paramTuple = std::get<1>(info.param);
  auto param = std::get<0>(paramTuple);

  auto queueMode = std::get<1>(paramTuple);

  std::stringstream test_name;
  test_name << platform_device_name << "__size__" << param.size
            << "__patternSize__" << param.pattern_size << "__" << queueMode;

  return test_name.str();
}

// Similar to urMultiDeviceContextTestWithParam this fixture allows a min
// device count to be specified, but in this case we duplicate existing
// devices to reach the min device count rather than skipping if it isn't met.
template <size_t minDevices>
struct urMultiQueueMultiDeviceTest : uur::urAllDevicesTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(uur::urAllDevicesTest::SetUp());

    ASSERT_SUCCESS(
        urContextCreate(devices.size(), devices.data(), nullptr, &context));

    // Duplicate our devices until we hit the minimum size specified.
    auto srcDevices = devices;
    while (devices.size() < minDevices) {
      devices.insert(devices.end(), srcDevices.begin(), srcDevices.end());
    }

    for (auto &device : devices) {
      ur_queue_handle_t queue = nullptr;
      ASSERT_SUCCESS(urQueueCreate(context, device, nullptr, &queue));
      queues.push_back(queue);
    }
  }

  void TearDown() override {
    for (auto &queue : queues) {
      EXPECT_SUCCESS(urQueueRelease(queue));
    }
    UUR_RETURN_ON_FATAL_FAILURE(uur::urAllDevicesTest::TearDown());
  }

  ur_context_handle_t context;
  std::vector<ur_queue_handle_t> queues;
};

template <size_t minDevices, class T>
struct urMultiQueueMultiDeviceTestWithParam
    : uur::urAllDevicesTestWithParam<T> {
  using uur::urAllDevicesTestWithParam<T>::devices;
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(uur::urAllDevicesTestWithParam<T>::SetUp());

    ASSERT_SUCCESS(
        urContextCreate(devices.size(), devices.data(), nullptr, &context));

    // Duplicate our devices until we hit the minimum size specified.
    std::vector<ur_device_handle_t> srcDevices;
    // If the test actually only wants one device duplicated a bunch of times
    // we take devices[0] and discard any other devices that were discovered.
    if (trueMultiDevice) {
      srcDevices = devices;
    } else {
      srcDevices.push_back(devices[0]);
      devices.clear();
    }
    while (devices.size() < minDevices) {
      devices.insert(devices.end(), srcDevices.begin(), srcDevices.end());
    }

    for (auto &device : devices) {
      ur_queue_handle_t queue = nullptr;
      ASSERT_SUCCESS(urQueueCreate(context, device, nullptr, &queue));
      queues.push_back(queue);
    }
  }

  void TearDown() override {
    for (auto &queue : queues) {
      EXPECT_SUCCESS(urQueueRelease(queue));
    }
    UUR_RETURN_ON_FATAL_FAILURE(uur::urAllDevicesTestWithParam<T>::TearDown());
  }

  ur_context_handle_t context;
  std::vector<ur_queue_handle_t> queues;
  bool trueMultiDevice = true;
};

} // namespace uur

#endif // UUR_ENQUEUE_RECT_HELPERS_H_INCLUDED
