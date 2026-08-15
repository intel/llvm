// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// UR reproducer for SYCL e2e test:
// sycl/test-e2e/DeprecatedFeatures/DiscardEvents/discard_events_l0_inorder.cpp
//
// RUN: %with-v1 env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 UR_L0_BATCH_SIZE=0 ./discard_events_in_order-test
// RUN: %with-v1 env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 UR_L0_BATCH_SIZE=1 ./discard_events_in_order-test
// RUN: %with-v1 env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 UR_L0_BATCH_SIZE=2 ./discard_events_in_order-test
// RUN: %with-v1 env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 UR_L0_BATCH_SIZE=3 ./discard_events_in_order-test

#include "uur/fixtures.h"
#include "uur/known_failure.h"
#include "uur/utils.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <string>
#include <vector>

namespace {

enum class UsmAllocType { Host, Shared, Device };

ur_exp_kernel_arg_properties_t makePointerArg(uint32_t index, const void *ptr) {
  ur_exp_kernel_arg_value_t arg_val = {};
  arg_val.pointer = ptr;
  return ur_exp_kernel_arg_properties_t{
      UR_STRUCTURE_TYPE_EXP_KERNEL_ARG_PROPERTIES, nullptr,
      UR_EXP_KERNEL_ARG_TYPE_POINTER, index, sizeof(void *), arg_val};
}

void launchKernel(ur_queue_handle_t queue, ur_kernel_handle_t kernel,
                  const void *values1, const void *values2,
          const void *values3, size_t buffer_size) {
  size_t global_offset = 0;
  size_t n_dimensions = 1;
  ur_exp_kernel_arg_properties_t args[3] = {
      makePointerArg(0, values1), makePointerArg(1, values2),
      makePointerArg(2, values3)};
  ASSERT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
      queue, kernel, n_dimensions, &global_offset, &buffer_size, nullptr, 3,
    args, nullptr, 0, nullptr, nullptr));
}

ur_result_t allocUsm(ur_context_handle_t context, ur_device_handle_t device,
                     UsmAllocType type, size_t size, void **ptr) {
  switch (type) {
  case UsmAllocType::Host:
    return urUSMHostAlloc(context, nullptr, nullptr, size, ptr);
  case UsmAllocType::Shared:
    return urUSMSharedAlloc(context, device, nullptr, nullptr, size, ptr);
  case UsmAllocType::Device:
    return urUSMDeviceAlloc(context, device, nullptr, nullptr, size, ptr);
  }
  return UR_RESULT_ERROR_INVALID_VALUE;
}

bool deviceSupportsUsm(ur_device_handle_t device, UsmAllocType type) {
  ur_device_usm_access_capability_flags_t flags = 0;
  ur_result_t result = UR_RESULT_ERROR_INVALID_VALUE;
  switch (type) {
  case UsmAllocType::Host:
    result = uur::GetDeviceUSMHostSupport(device, flags);
    break;
  case UsmAllocType::Shared:
    result = uur::GetDeviceUSMSingleSharedSupport(device, flags);
    break;
  case UsmAllocType::Device:
    result = uur::GetDeviceUSMDeviceSupport(device, flags);
    break;
  }
  if (result != UR_RESULT_SUCCESS)
    return false;
  return (flags & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS) != 0;
}

void enqueueMemcpy(ur_queue_handle_t queue, void *dst, const void *src,
                   size_t size) {
  ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, dst, const_cast<void *>(src),
                                    size, 0, nullptr, nullptr));
}

void enqueueFill(ur_queue_handle_t queue, void *ptr, const void *pattern,
                 size_t patternSize, size_t size) {
  ASSERT_SUCCESS(
      urEnqueueUSMFill(queue, ptr, patternSize, pattern, size, 0, nullptr,
                       nullptr));
}

void enqueueKernel(ur_queue_handle_t queue, ur_kernel_handle_t kernel,
                   const void *values1,
                   const void *values2, const void *values3,
                   size_t buffer_size) {
  launchKernel(queue, kernel, values1, values2, values3, buffer_size);
}

void runCalculation(ur_queue_handle_t queue, ur_kernel_handle_t kernels[5],
                    ur_context_handle_t context, ur_device_handle_t device,
                    UsmAllocType allocType, size_t buffer_size,
                    size_t byte_size) {
  int *values1 = nullptr;
  int *values2 = nullptr;
  int *values3 = nullptr;
  ASSERT_SUCCESS(allocUsm(context, device, allocType, byte_size,
                          reinterpret_cast<void **>(&values1)));
  ASSERT_SUCCESS(allocUsm(context, device, allocType, byte_size,
                          reinterpret_cast<void **>(&values2)));
  ASSERT_SUCCESS(allocUsm(context, device, allocType, byte_size,
                          reinterpret_cast<void **>(&values3)));

  std::vector<int> values(buffer_size, 0);
  std::iota(values.begin(), values.end(), 0);
  std::vector<int> vec1(buffer_size, 0);
  std::vector<int> vec2(buffer_size, 0);
  std::vector<int> vec3(buffer_size, 0);

  enqueueMemcpy(queue, values1, values.data(), byte_size);
  enqueueMemcpy(queue, values2, values1, byte_size);
  enqueueMemcpy(queue, values3, values2, byte_size);

  int zero = 0;
  enqueueFill(queue, values1, &zero, sizeof(zero), byte_size);

  enqueueKernel(queue, kernels[0], values1, values2, values3, buffer_size);
  enqueueKernel(queue, kernels[1], values1, values2, values3, buffer_size);

  enqueueMemcpy(queue, values.data(), values1, byte_size);
  enqueueMemcpy(queue, values2, values.data(), byte_size);

  enqueueKernel(queue, kernels[2], values1, values2, values3, buffer_size);
  enqueueKernel(queue, kernels[3], values1, values2, values3, buffer_size);
  enqueueKernel(queue, kernels[4], values1, values2, values3, buffer_size);

  enqueueMemcpy(queue, vec1.data(), values1, byte_size);
  enqueueMemcpy(queue, vec2.data(), values2, byte_size);
  enqueueMemcpy(queue, vec3.data(), values3, byte_size);

  ASSERT_SUCCESS(urQueueFinish(queue));

  for (size_t i = 0; i < buffer_size; ++i) {
    ASSERT_EQ(vec1[i], static_cast<int>(i) + 11110);
    ASSERT_EQ(vec2[i], static_cast<int>(i));
    ASSERT_EQ(vec3[i], static_cast<int>(i));
  }

  ASSERT_SUCCESS(urUSMFree(context, values1));
  ASSERT_SUCCESS(urUSMFree(context, values2));
  ASSERT_SUCCESS(urUSMFree(context, values3));
}

} // namespace

struct urDiscardEventsInOrderTest : uur::urContextTest {
  static constexpr char ProgramName[] = "discard_events_in_order_kernels";
  static constexpr size_t BufferSize = 100;
  static constexpr size_t ByteSize = BufferSize * sizeof(int);

  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(uur::urContextTest::SetUp());

    if (uur::detail::getAdapterInfo(adapter).backend != UR_BACKEND_LEVEL_ZERO) {
      GTEST_SKIP() << "Test requires Level Zero adapter";
    }

    ur_queue_properties_t props = {UR_STRUCTURE_TYPE_QUEUE_PROPERTIES, nullptr,
                                   UR_QUEUE_FLAG_DISCARD_EVENTS};
    ASSERT_SUCCESS(urQueueCreate(context, device, &props, &queue));
    ASSERT_NE(queue, nullptr);

    std::shared_ptr<std::vector<char>> il_binary;
    UUR_RETURN_ON_FATAL_FAILURE(uur::KernelsEnvironment::instance->LoadSource(
        ProgramName, platform, il_binary));

    const ur_program_properties_t properties = {
        UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES, nullptr, 0, nullptr};
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::KernelsEnvironment::instance->CreateProgram(
            platform, context, device, *il_binary, &properties, &program));
    ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));

    const auto kernelNames =
        uur::KernelsEnvironment::instance->GetEntryPointNames(ProgramName);
    const std::array<const char *, 5> kernelTags = {
        "DiscardEventsK1", "DiscardEventsK2", "DiscardEventsK3", "DiscardEventsK4",
        "DiscardEventsK5"};
    for (size_t i = 0; i < kernelTags.size(); ++i) {
      auto it = std::find_if(
          kernelNames.begin(), kernelNames.end(), [&](const std::string &name) {
            return name.find(kernelTags[i]) != std::string::npos;
          });
      ASSERT_NE(it, kernelNames.end())
          << "kernel " << kernelTags[i] << " not in device binary (have "
          << kernelNames.size() << " entry points)";
      ASSERT_SUCCESS(urKernelCreate(program, it->c_str(), &kernels[i]));
    }
  }

  void TearDown() override {
    for (auto &k : kernels) {
      if (k) {
        EXPECT_SUCCESS(urKernelRelease(k));
      }
    }
    if (program) {
      EXPECT_SUCCESS(urProgramRelease(program));
    }
    if (queue) {
      EXPECT_SUCCESS(urQueueRelease(queue));
    }
    UUR_RETURN_ON_FATAL_FAILURE(uur::urContextTest::TearDown());
  }

  void runForUsm(UsmAllocType allocType) {
    if (!deviceSupportsUsm(device, allocType)) {
      GTEST_SKIP() << "Selected USM type is not supported";
    }
    runCalculation(queue, kernels, context, device, allocType, BufferSize,
                   ByteSize);
  }

  ur_queue_handle_t queue = nullptr;
  ur_program_handle_t program = nullptr;
  ur_kernel_handle_t kernels[5] = {};
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urDiscardEventsInOrderTest);

TEST_P(urDiscardEventsInOrderTest, HostUsm) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});
  runForUsm(UsmAllocType::Host);
}

TEST_P(urDiscardEventsInOrderTest, SharedUsm) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});
  runForUsm(UsmAllocType::Shared);
}

TEST_P(urDiscardEventsInOrderTest, DeviceUsm) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});
  runForUsm(UsmAllocType::Device);
}



