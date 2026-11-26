// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %with-v2 ./memory_alloc_device-test
// REQUIRES: v2

#include "level_zero/common/device.hpp"
#include "level_zero/v2/memory.hpp"

#include "uur/checks.h"
#include "uur/fixtures.h"
#include "uur/raii.h"
#include "uur/utils.h"
#include "ze_api.h"

#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <numeric>

namespace v2 = ur::level_zero::v2;

// Tests that ur_mem_alloc_properties_t places the memory object's storage on
// the requested device rather than on the first device of the context.
struct urMemAllocDeviceTest : uur::urMultiDeviceContextTestTemplate<1> {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::urMultiDeviceContextTestTemplate<1>::SetUp());

    // Default-constructed queues pick the first device, so use the last one to
    // make sure the request is honored. With a single device, that is the one.
    targetDevice = devices.back();
    allocProps = {UR_STRUCTURE_TYPE_MEM_ALLOC_PROPERTIES, nullptr,
                  targetDevice};
  }

  // Returns the device that owns the storage of a buffer.
  ze_device_handle_t getBufferZeDevice(ur_mem_handle_t buffer) {
    ur_native_handle_t nativeMem = 0;
    // A null device makes the adapter report the existing allocation instead
    // of migrating the buffer.
    EXPECT_SUCCESS(urMemGetNativeHandle(buffer, nullptr, &nativeMem));

    ur_native_handle_t nativeContext = 0;
    EXPECT_SUCCESS(urContextGetNativeHandle(context, &nativeContext));

    ze_memory_allocation_properties_t props = {};
    props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    ze_device_handle_t zeDevice = nullptr;
    EXPECT_EQ(ZE_RESULT_SUCCESS,
              zeMemGetAllocProperties(
                  reinterpret_cast<ze_context_handle_t>(nativeContext),
                  reinterpret_cast<void *>(nativeMem), &props, &zeDevice));
    EXPECT_EQ(ZE_MEMORY_TYPE_DEVICE, props.type);
    return zeDevice;
  }

  ze_device_handle_t getTargetZeDevice() {
    ur_native_handle_t nativeDevice = 0;
    EXPECT_SUCCESS(urDeviceGetNativeHandle(targetDevice, &nativeDevice));
    return reinterpret_cast<ze_device_handle_t>(nativeDevice);
  }

  ur_device_handle_t targetDevice = nullptr;
  ur_mem_alloc_properties_t allocProps = {};
};

UUR_INSTANTIATE_PLATFORM_TEST_SUITE(urMemAllocDeviceTest);

TEST_P(urMemAllocDeviceTest, BufferIsAllocatedOnRequestedDevice) {
  // Integrated devices use host memory for buffers, which has no owner.
  ur_bool_t integrated = false;
  ASSERT_SUCCESS(urDeviceGetInfo(targetDevice, UR_DEVICE_INFO_IS_INTEGRATED_GPU,
                                 sizeof(integrated), &integrated, nullptr));
  if (devices.size() == 1 && integrated) {
    GTEST_SKIP() << "Integrated device buffers are allocated in host memory";
  }

  constexpr size_t size = 4096;
  ur_buffer_properties_t props = {UR_STRUCTURE_TYPE_BUFFER_PROPERTIES,
                                  &allocProps, nullptr};

  uur::raii::Mem buffer = nullptr;
  ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, size,
                                   &props, buffer.ptr()));
  ASSERT_NE(nullptr, buffer.get());

  ASSERT_EQ(getTargetZeDevice(), getBufferZeDevice(buffer.get()));
}

TEST_P(urMemAllocDeviceTest, BufferWithHostDataIsAllocatedOnRequestedDevice) {
  ur_bool_t integrated = false;
  ASSERT_SUCCESS(urDeviceGetInfo(targetDevice, UR_DEVICE_INFO_IS_INTEGRATED_GPU,
                                 sizeof(integrated), &integrated, nullptr));
  if (devices.size() == 1 && integrated) {
    GTEST_SKIP() << "Integrated device buffers are allocated in host memory";
  }

  constexpr size_t count = 1024;
  std::array<uint32_t, count> input;
  std::iota(input.begin(), input.end(), 0);

  ur_buffer_properties_t props = {UR_STRUCTURE_TYPE_BUFFER_PROPERTIES,
                                  &allocProps, input.data()};

  uur::raii::Mem buffer = nullptr;
  ASSERT_SUCCESS(urMemBufferCreate(
      context, UR_MEM_FLAG_READ_WRITE | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER,
      sizeof(input), &props, buffer.ptr()));
  ASSERT_NE(nullptr, buffer.get());

  ASSERT_EQ(getTargetZeDevice(), getBufferZeDevice(buffer.get()));

  // The host data must have been copied to the requested device.
  uur::raii::Queue queue = nullptr;
  ASSERT_SUCCESS(urQueueCreate(context, targetDevice, nullptr, queue.ptr()));
  std::array<uint32_t, count> output = {};
  ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, buffer.get(), true, 0,
                                        sizeof(output), output.data(), 0,
                                        nullptr, nullptr));
  ASSERT_EQ(input, output);
}

TEST_P(urMemAllocDeviceTest, ImageIsAllocatedOnRequestedDevice) {
  bool imageSupport = false;
  ASSERT_SUCCESS(uur::GetDeviceImageSupport(targetDevice, imageSupport));
  if (!imageSupport) {
    GTEST_SKIP() << "Device doesn't support images";
  }

  ur_image_format_t format = {UR_IMAGE_CHANNEL_ORDER_RGBA,
                              UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32};
  ur_image_desc_t desc = {UR_STRUCTURE_TYPE_IMAGE_DESC,
                          &allocProps,
                          UR_MEM_TYPE_IMAGE2D,
                          4, // width
                          4, // height
                          0, // depth
                          1, // array size
                          0, // row pitch
                          0, // slice pitch
                          0, // MIP levels
                          0};

  uur::raii::Mem image = nullptr;
  ASSERT_SUCCESS(urMemImageCreate(context, UR_MEM_FLAG_READ_WRITE, &format,
                                  &desc, nullptr, image.ptr()));
  ASSERT_NE(nullptr, image.get());

  auto *internalImage = v2::v2_cast(image.get())->getImage();
  ASSERT_EQ(ur::level_zero::common_cast(targetDevice),
            internalImage->getDevice());
}
