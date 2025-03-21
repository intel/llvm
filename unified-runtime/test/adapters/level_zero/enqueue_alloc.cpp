// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thread>

#include "ur_api.h"
#include <uur/fixtures.h>

struct EnqueueAllocTestParam {
  ur_result_t (*enqueueUSMAllocFunc)(
      ur_queue_handle_t, ur_usm_pool_handle_t, const size_t,
      const ur_exp_async_usm_alloc_properties_t *, uint32_t,
      const ur_event_handle_t *, void **, ur_event_handle_t *);
  ur_result_t (*checkUSMSupportFunc)(ur_device_handle_t,
                                     ur_device_usm_access_capability_flags_t &);
};

std::ostream &operator<<(std::ostream &os, EnqueueAllocTestParam param) {
  if (param.enqueueUSMAllocFunc == urEnqueueUSMHostAllocExp) {
    os << " urEnqueueUSMHostAllocExp";
  } else if (param.enqueueUSMAllocFunc == urEnqueueUSMDeviceAllocExp) {
    os << " urEnqueueUSMDeviceAllocExp";
  } else if (param.enqueueUSMAllocFunc == urEnqueueUSMSharedAllocExp) {
    os << " urEnqueueUSMSharedAllocExp";
  }

  return os;
}

struct urL0EnqueueAllocTest
    : uur::urKernelExecutionTestWithParam<EnqueueAllocTestParam> {
  void SetUp() override {
    program_name = "fill_usm";
    UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTestWithParam::SetUp());
  }

  void ValidateEnqueueFree(void *ptr) {
    ur_event_handle_t freeEvent = nullptr;
    ASSERT_NE(ptr, nullptr);
    ASSERT_SUCCESS(
        urEnqueueUSMFreeExp(queue, nullptr, ptr, 0, nullptr, &freeEvent));
    ASSERT_NE(freeEvent, nullptr);
    ASSERT_SUCCESS(urQueueFinish(queue));
    ASSERT_SUCCESS(urEventRelease(freeEvent));
  }

  const size_t ARRAY_SIZE = 16;
  const uint32_t DATA = 0xC0FFEE;
};

struct EnqueueAllocMultiQueueTestParam {
  size_t allocSize;
  size_t iterations;
  size_t numQueues;
  EnqueueAllocTestParam funcParams;
};

std::ostream &operator<<(std::ostream &os,
                         EnqueueAllocMultiQueueTestParam param) {
  os << "alloc size " << param.allocSize;
  os << " queues " << param.numQueues;
  os << " iterations " << param.iterations;
  os << " " << param.funcParams;

  return os;
}

struct urL0EnqueueAllocMultiQueueSameDeviceTest
    : uur::urContextTestWithParam<EnqueueAllocMultiQueueTestParam> {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam::SetUp());
    auto param = std::get<1>(this->GetParam());

    queues.reserve(param.numQueues);
    for (size_t i = 0; i < param.numQueues; i++) {
      ur_queue_handle_t queue = nullptr;
      ASSERT_SUCCESS(urQueueCreate(context, device, 0, &queue));
      queues.push_back(queue);
    }
  }

  void TearDown() override {
    for (auto &queue : queues) {
      EXPECT_SUCCESS(urQueueRelease(queue));
    }

    UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam::TearDown());
  }

  std::vector<ur_queue_handle_t> queues;
};

struct EnqueueAllocMultiQueueMultiDeviceTestParam {
  size_t allocSize;
  size_t iterations;
  EnqueueAllocTestParam funcParams;
};

std::ostream &operator<<(std::ostream &os,
                         EnqueueAllocMultiQueueMultiDeviceTestParam param) {
  os << "alloc size " << param.allocSize;
  os << " iterations " << param.iterations;
  os << " " << param.funcParams;

  return os;
}

struct urL0EnqueueAllocMultiQueueMultiDeviceTest
    : public uur::urAllDevicesTestWithParam<
          EnqueueAllocMultiQueueMultiDeviceTestParam> {
  using uur::urAllDevicesTestWithParam<
      EnqueueAllocMultiQueueMultiDeviceTestParam>::devices;
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::urAllDevicesTestWithParam<
            EnqueueAllocMultiQueueMultiDeviceTestParam>::SetUp());

    if (devices.size() < 2) {
      GTEST_SKIP() << "Need at least 2 devices, found " << devices.size();
    }

    ASSERT_SUCCESS(
        urContextCreate(devices.size(), devices.data(), nullptr, &context));

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
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::urAllDevicesTestWithParam<
            EnqueueAllocMultiQueueMultiDeviceTestParam>::TearDown());
  }

  void ValidateFill(void *ptr, uint8_t pattern, size_t size) {
    // Copy the allocation to a host one so we can validate the
    void *hostAlloc = nullptr;
    ASSERT_SUCCESS(urUSMHostAlloc(context, nullptr, nullptr, size, &hostAlloc));
    ASSERT_NE(hostAlloc, nullptr);
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queues[0], true, hostAlloc, ptr, size, 0,
                                      nullptr, nullptr));
    std::ignore = pattern;
    for (size_t i = 0; i * sizeof(uint8_t) < size; i++) {
      ASSERT_EQ(*static_cast<uint8_t *>(hostAlloc), pattern);
    }
  }

  ur_context_handle_t context;
  std::vector<ur_queue_handle_t> queues;
};

UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urL0EnqueueAllocTest,
    ::testing::ValuesIn({
        EnqueueAllocTestParam{urEnqueueUSMHostAllocExp,
                              uur::GetDeviceUSMHostSupport},
        EnqueueAllocTestParam{urEnqueueUSMSharedAllocExp,
                              uur::GetDeviceUSMSingleSharedSupport},
        EnqueueAllocTestParam{urEnqueueUSMDeviceAllocExp,
                              uur::GetDeviceUSMDeviceSupport},
    }),
    uur::deviceTestWithParamPrinter<EnqueueAllocTestParam>);

TEST_P(urL0EnqueueAllocTest, Success) {
  const auto enqueueUSMAllocFunc =
      std::get<1>(this->GetParam()).enqueueUSMAllocFunc;
  const auto checkUSMSupportFunc =
      std::get<1>(this->GetParam()).checkUSMSupportFunc;

  ur_device_usm_access_capability_flags_t USMSupport = 0;
  ASSERT_SUCCESS(checkUSMSupportFunc(device, USMSupport));
  if (!(USMSupport & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
    GTEST_SKIP() << "Selected USM type is not supported.";
  }

  void *ptr = nullptr;
  ur_event_handle_t allocEvent = nullptr;
  ASSERT_SUCCESS(enqueueUSMAllocFunc(queue, nullptr, sizeof(DATA), nullptr, 0,
                                     nullptr, &ptr, &allocEvent));
  ASSERT_NE(ptr, nullptr);
  ASSERT_NE(allocEvent, nullptr);

  // Access the allocation.
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(DATA), &DATA, sizeof(DATA),
                                  1, &allocEvent, nullptr));

  ValidateEnqueueFree(ptr);
  ASSERT_SUCCESS(urEventRelease(allocEvent));
}

TEST_P(urL0EnqueueAllocTest, SuccessReuse) {
  const auto enqueueUSMAllocFunc =
      std::get<1>(this->GetParam()).enqueueUSMAllocFunc;
  const auto checkUSMSupportFunc =
      std::get<1>(this->GetParam()).checkUSMSupportFunc;

  ur_device_usm_access_capability_flags_t USMSupport = 0;
  ASSERT_SUCCESS(checkUSMSupportFunc(device, USMSupport));
  if (!(USMSupport & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
    GTEST_SKIP() << "Selected USM type is not supported.";
  }

  void *ptr = nullptr;
  ur_event_handle_t allocEvent = nullptr;
  ASSERT_SUCCESS(enqueueUSMAllocFunc(queue, nullptr, sizeof(DATA), nullptr, 0,
                                     nullptr, &ptr, &allocEvent));
  ASSERT_NE(ptr, nullptr);
  ASSERT_NE(allocEvent, nullptr);

  ur_event_handle_t freeEvent = nullptr;
  ASSERT_SUCCESS(
      urEnqueueUSMFreeExp(queue, nullptr, ptr, 1, &allocEvent, &freeEvent));
  ASSERT_NE(freeEvent, nullptr);

  void *ptr2 = nullptr;
  ur_event_handle_t allocEvent2 = nullptr;
  ASSERT_SUCCESS(enqueueUSMAllocFunc(queue, nullptr, sizeof(DATA), nullptr, 1,
                                     &freeEvent, &ptr2, &allocEvent2));
  ASSERT_EQ(ptr2, ptr); // Memory should be reused from previous allocation.
  ASSERT_NE(allocEvent2, nullptr);

  ASSERT_SUCCESS(urEventRelease(allocEvent));
  ASSERT_SUCCESS(urEventRelease(freeEvent));
  ASSERT_SUCCESS(urEventRelease(allocEvent2));

  ValidateEnqueueFree(ptr2);
}

TEST_P(urL0EnqueueAllocTest, SuccessFromPool) {
  const auto enqueueUSMAllocFunc =
      std::get<1>(this->GetParam()).enqueueUSMAllocFunc;
  const auto checkUSMSupportFunc =
      std::get<1>(this->GetParam()).checkUSMSupportFunc;

  ur_device_usm_access_capability_flags_t USMSupport = 0;
  ASSERT_SUCCESS(checkUSMSupportFunc(device, USMSupport));
  if (!(USMSupport & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
    GTEST_SKIP() << "Selected USM type is not supported.";
  }

  ur_usm_pool_handle_t pool = nullptr;
  ur_usm_pool_desc_t desc = {UR_STRUCTURE_TYPE_USM_POOL_DESC, nullptr, 0};
  ASSERT_SUCCESS(urUSMPoolCreate(context, &desc, &pool));

  void *ptr = nullptr;
  ur_event_handle_t allocEvent = nullptr;
  ASSERT_SUCCESS(enqueueUSMAllocFunc(queue, pool, sizeof(uint32_t), nullptr, 0,
                                     nullptr, &ptr, &allocEvent));
  ASSERT_NE(ptr, nullptr);
  ASSERT_NE(allocEvent, nullptr);

  // Access the allocation.
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(DATA), &DATA, sizeof(DATA),
                                  1, &allocEvent, nullptr));

  ValidateEnqueueFree(ptr);
  ASSERT_SUCCESS(urEventRelease(allocEvent));
}

TEST_P(urL0EnqueueAllocTest, SuccessWithKernel) {
  const auto enqueueUSMAllocFunc =
      std::get<1>(this->GetParam()).enqueueUSMAllocFunc;
  const auto checkUSMSupportFunc =
      std::get<1>(this->GetParam()).checkUSMSupportFunc;

  ur_device_usm_access_capability_flags_t USMSupport = 0;
  ASSERT_SUCCESS(checkUSMSupportFunc(device, USMSupport));
  if (!(USMSupport & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
    GTEST_SKIP() << "Selected USM type is not supported.";
  }

  void *ptr = nullptr;
  ASSERT_SUCCESS(enqueueUSMAllocFunc(queue, nullptr,
                                     ARRAY_SIZE * sizeof(uint32_t), nullptr, 0,
                                     nullptr, &ptr, nullptr));
  ASSERT_NE(ptr, nullptr);

  ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, ptr));
  ASSERT_SUCCESS(urKernelSetArgValue(kernel, 1, sizeof(DATA), nullptr, &DATA));
  Launch1DRange(ARRAY_SIZE);

  ValidateEnqueueFree(ptr);
}

TEST_P(urL0EnqueueAllocTest, SuccessWithKernelRepeat) {
  const auto enqueueUSMAllocFunc =
      std::get<1>(this->GetParam()).enqueueUSMAllocFunc;
  const auto checkUSMSupportFunc =
      std::get<1>(this->GetParam()).checkUSMSupportFunc;

  ur_device_usm_access_capability_flags_t USMSupport = 0;
  ASSERT_SUCCESS(checkUSMSupportFunc(device, USMSupport));
  if (!(USMSupport & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
    GTEST_SKIP() << "Selected USM type is not supported.";
  }

  void *ptr = nullptr;
  ASSERT_SUCCESS(enqueueUSMAllocFunc(queue, nullptr,
                                     ARRAY_SIZE * sizeof(uint32_t), nullptr, 0,
                                     nullptr, &ptr, nullptr));
  ASSERT_NE(ptr, nullptr);

  ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, ptr));
  ASSERT_SUCCESS(urKernelSetArgValue(kernel, 1, sizeof(DATA), nullptr, &DATA));
  Launch1DRange(ARRAY_SIZE);

  ASSERT_SUCCESS(urEnqueueUSMFreeExp(queue, nullptr, ptr, 0, nullptr, nullptr));

  void *ptr2 = nullptr;
  ASSERT_SUCCESS(enqueueUSMAllocFunc(queue, nullptr,
                                     ARRAY_SIZE * sizeof(uint32_t), nullptr, 0,
                                     nullptr, &ptr2, nullptr));
  ASSERT_NE(ptr2, nullptr);

  ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, ptr2));
  ASSERT_SUCCESS(urKernelSetArgValue(kernel, 1, sizeof(DATA), nullptr, &DATA));
  Launch1DRange(ARRAY_SIZE);

  ValidateEnqueueFree(ptr2);
}

UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urL0EnqueueAllocMultiQueueSameDeviceTest,
    ::testing::ValuesIn({
        EnqueueAllocMultiQueueTestParam{1024, 256, 8, urEnqueueUSMHostAllocExp,
                                        uur::GetDeviceUSMHostSupport},
        EnqueueAllocMultiQueueTestParam{1024, 256, 8,
                                        urEnqueueUSMSharedAllocExp,
                                        uur::GetDeviceUSMSingleSharedSupport},
        EnqueueAllocMultiQueueTestParam{1024, 256, 8,
                                        urEnqueueUSMDeviceAllocExp,
                                        uur::GetDeviceUSMDeviceSupport},
    }),
    uur::deviceTestWithParamPrinter<EnqueueAllocMultiQueueTestParam>);

TEST_P(urL0EnqueueAllocMultiQueueSameDeviceTest, SuccessMt) {
  const size_t allocSize = std::get<1>(this->GetParam()).allocSize;
  const size_t numQueues = std::get<1>(this->GetParam()).numQueues;
  const size_t iterations = std::get<1>(this->GetParam()).iterations;
  const auto enqueueUSMAllocFunc =
      std::get<1>(this->GetParam()).funcParams.enqueueUSMAllocFunc;
  const auto checkUSMSupportFunc =
      std::get<1>(this->GetParam()).funcParams.checkUSMSupportFunc;

  ur_device_usm_access_capability_flags_t USMSupport = 0;
  ASSERT_SUCCESS(checkUSMSupportFunc(device, USMSupport));
  if (!(USMSupport & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
    GTEST_SKIP() << "Selected USM type is not supported.";
  }

  std::vector<std::thread> threads;
  for (size_t idx = 0; idx < numQueues; idx++) {
    // Create a thread per queue.
    threads.emplace_back([&, idx, iterations] {
      std::vector<void *> ptrs(iterations);
      for (size_t i = 0; i < iterations; i++) {
        ASSERT_SUCCESS(enqueueUSMAllocFunc(queues[idx], nullptr, allocSize,
                                           nullptr, 0, nullptr, &ptrs[i],
                                           nullptr));
        ASSERT_NE(ptrs[i], nullptr);
      }

      ASSERT_SUCCESS(urQueueFinish(queues[idx]));

      uint8_t fillPattern = 0xAF;
      for (size_t i = 0; i < iterations; i++) {
        ASSERT_SUCCESS(urEnqueueUSMFill(queues[idx], ptrs[i], 1, &fillPattern,
                                        allocSize, 0, nullptr, nullptr));
      }

      ASSERT_SUCCESS(urQueueFinish(queues[idx]));

      for (size_t i = 0; i < iterations; i++) {
        ASSERT_SUCCESS(urEnqueueUSMFreeExp(queues[idx], nullptr, ptrs[i], 0,
                                           nullptr, nullptr));
      }

      ASSERT_SUCCESS(urQueueFinish(queues[idx]));
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }
}

TEST_P(urL0EnqueueAllocMultiQueueSameDeviceTest, SuccessReuse) {
  const size_t allocSize = std::get<1>(this->GetParam()).allocSize;
  const auto enqueueUSMAllocFunc =
      std::get<1>(this->GetParam()).funcParams.enqueueUSMAllocFunc;
  const auto checkUSMSupportFunc =
      std::get<1>(this->GetParam()).funcParams.checkUSMSupportFunc;

  ur_device_usm_access_capability_flags_t USMSupport = 0;
  ASSERT_SUCCESS(checkUSMSupportFunc(device, USMSupport));
  if (!(USMSupport & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
    GTEST_SKIP() << "Selected USM type is not supported.";
  }

  std::vector<void *> ptrs(queues.size());
  std::vector<ur_event_handle_t> allocEvents(queues.size());
  std::vector<ur_event_handle_t> fillEvents(queues.size());
  std::vector<ur_event_handle_t> freeEvents(queues.size());

  ASSERT_SUCCESS(enqueueUSMAllocFunc(queues[0], nullptr, allocSize, nullptr, 0,
                                     nullptr, &ptrs[0], &allocEvents[0]));
  ASSERT_NE(ptrs[0], nullptr);
  ASSERT_NE(allocEvents[0], nullptr);

  uint8_t fillPattern = 0xAF;
  ASSERT_SUCCESS(urEnqueueUSMFill(queues[0], ptrs[0], 1, &fillPattern,
                                  allocSize, 1, &allocEvents[0],
                                  &fillEvents[0]));
  ASSERT_NE(fillEvents[0], nullptr);

  ASSERT_SUCCESS(urEnqueueUSMFreeExp(queues[0], nullptr, ptrs[0], 1,
                                     &fillEvents[0], &freeEvents[0]));
  ASSERT_NE(freeEvents[0], nullptr);

  for (size_t qIdx = 1; qIdx < queues.size(); qIdx++) {
    ASSERT_SUCCESS(enqueueUSMAllocFunc(queues[qIdx], nullptr, allocSize,
                                       nullptr, 1, &freeEvents[qIdx - 1],
                                       &ptrs[qIdx], &allocEvents[qIdx]));
    ASSERT_EQ(
        ptrs[qIdx],
        ptrs[qIdx - 1]); // Memory should be reused from previous allocation.
    ASSERT_NE(allocEvents[qIdx], nullptr);

    // Access the memory.
    ASSERT_SUCCESS(urEnqueueUSMFill(queues[qIdx], ptrs[qIdx], 1, &fillPattern,
                                    allocSize, 1, &allocEvents[qIdx],
                                    &fillEvents[qIdx]));
    ASSERT_NE(fillEvents[qIdx], nullptr);

    ASSERT_SUCCESS(urEnqueueUSMFreeExp(queues[qIdx], nullptr, ptrs[qIdx], 1,
                                       &fillEvents[qIdx], &freeEvents[qIdx]));
    ASSERT_NE(freeEvents[qIdx], nullptr);
  }

  // Wait for the queue that received the last event.
  ASSERT_SUCCESS(urQueueFinish(queues[queues.size() - 1]));

  for (size_t i = 0; i < queues.size(); i++) {
    ASSERT_SUCCESS(urEventRelease(allocEvents[i]));
    ASSERT_SUCCESS(urEventRelease(fillEvents[i]));
    ASSERT_SUCCESS(urEventRelease(freeEvents[i]));
  }
}

TEST_P(urL0EnqueueAllocMultiQueueSameDeviceTest, SuccessDependantMt) {
  const size_t allocSize = std::get<1>(this->GetParam()).allocSize;
  const size_t iterations = std::get<1>(this->GetParam()).iterations;
  const auto enqueueUSMAllocFunc =
      std::get<1>(this->GetParam()).funcParams.enqueueUSMAllocFunc;
  const auto checkUSMSupportFunc =
      std::get<1>(this->GetParam()).funcParams.checkUSMSupportFunc;

  ur_device_usm_access_capability_flags_t USMSupport = 0;
  ASSERT_SUCCESS(checkUSMSupportFunc(device, USMSupport));
  if (!(USMSupport & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
    GTEST_SKIP() << "Selected USM type is not supported.";
  }

  std::vector<std::thread> threads;
  for (size_t qIdx = 0; qIdx < queues.size(); qIdx++) {
    // Create a thread per queue.
    threads.emplace_back([&, qIdx, iterations] {
      std::vector<void *> ptrs(iterations);
      std::vector<ur_event_handle_t> allocEvents(iterations);
      std::vector<ur_event_handle_t> fillEvents(iterations);
      std::vector<ur_event_handle_t> freeEvents(iterations);

      ASSERT_SUCCESS(enqueueUSMAllocFunc(queues[qIdx], nullptr, allocSize,
                                         nullptr, 0, nullptr, &ptrs[0],
                                         &allocEvents[0]));
      ASSERT_NE(ptrs[0], nullptr);
      ASSERT_NE(allocEvents[0], nullptr);

      uint8_t fillPattern = 0xAF;
      ASSERT_SUCCESS(urEnqueueUSMFill(queues[qIdx], ptrs[0], 1, &fillPattern,
                                      allocSize, 1, &allocEvents[0],
                                      &fillEvents[0]));
      ASSERT_NE(fillEvents[0], nullptr);

      ASSERT_SUCCESS(urEnqueueUSMFreeExp(queues[qIdx], nullptr, ptrs[0], 1,
                                         &fillEvents[0], &freeEvents[0]));
      ASSERT_NE(freeEvents[0], nullptr);

      for (size_t i = 1; i < iterations; i++) {
        ASSERT_SUCCESS(enqueueUSMAllocFunc(queues[qIdx], nullptr, allocSize,
                                           nullptr, 1, &freeEvents[i - 1],
                                           &ptrs[i], &allocEvents[i]));
        ASSERT_NE(ptrs[i], nullptr);
        ASSERT_NE(allocEvents[i], nullptr);

        // Access the memory.
        ASSERT_SUCCESS(urEnqueueUSMFill(queues[qIdx], ptrs[i], 1, &fillPattern,
                                        allocSize, 1, &allocEvents[i],
                                        &fillEvents[i]));
        ASSERT_NE(fillEvents[i], nullptr);

        ASSERT_SUCCESS(urEnqueueUSMFreeExp(queues[qIdx], nullptr, ptrs[i], 1,
                                           &fillEvents[i], &freeEvents[i]));
        ASSERT_NE(freeEvents[i], nullptr);
      }

      ASSERT_SUCCESS(urQueueFinish(queues[qIdx]));

      for (size_t i = 0; i < iterations; i++) {
        ASSERT_SUCCESS(urEventRelease(allocEvents[i]));
        ASSERT_SUCCESS(urEventRelease(fillEvents[i]));
        ASSERT_SUCCESS(urEventRelease(freeEvents[i]));
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }
}

UUR_PLATFORM_TEST_SUITE_WITH_PARAM(
    urL0EnqueueAllocMultiQueueMultiDeviceTest,
    ::testing::ValuesIn({
        EnqueueAllocMultiQueueMultiDeviceTestParam{
            1024, 256, urEnqueueUSMHostAllocExp, uur::GetDeviceUSMHostSupport},
        EnqueueAllocMultiQueueMultiDeviceTestParam{
            1024, 256, urEnqueueUSMSharedAllocExp,
            uur::GetDeviceUSMSingleSharedSupport},
        EnqueueAllocMultiQueueMultiDeviceTestParam{
            1024, 256, urEnqueueUSMDeviceAllocExp,
            uur::GetDeviceUSMDeviceSupport},
    }),
    uur::platformTestWithParamPrinter<
        EnqueueAllocMultiQueueMultiDeviceTestParam>);

TEST_P(urL0EnqueueAllocMultiQueueMultiDeviceTest, SuccessDependantMt) {
  const size_t allocSize = std::get<1>(this->GetParam()).allocSize;
  const size_t iterations = std::get<1>(this->GetParam()).iterations;
  const auto enqueueUSMAllocFunc =
      std::get<1>(this->GetParam()).funcParams.enqueueUSMAllocFunc;
  const auto checkUSMSupportFunc =
      std::get<1>(this->GetParam()).funcParams.checkUSMSupportFunc;

  ur_device_usm_access_capability_flags_t USMSupport = 0;
  for (auto &device : devices) {
    ASSERT_SUCCESS(checkUSMSupportFunc(device, USMSupport));
    if (!(USMSupport & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
      GTEST_SKIP() << "Selected USM type is not supported.";
    }
  }

  std::vector<std::thread> threads;
  for (size_t qIdx = 0; qIdx < queues.size(); qIdx++) {
    // Create a thread per queue.
    threads.emplace_back([&, qIdx, iterations] {
      std::vector<void *> ptrs(iterations);
      std::vector<ur_event_handle_t> allocEvents(iterations);
      std::vector<ur_event_handle_t> fillEvents(iterations);
      std::vector<ur_event_handle_t> freeEvents(iterations);

      ASSERT_SUCCESS(enqueueUSMAllocFunc(queues[qIdx], nullptr, allocSize,
                                         nullptr, 0, nullptr, &ptrs[0],
                                         &allocEvents[0]));
      ASSERT_NE(ptrs[0], nullptr);
      ASSERT_NE(allocEvents[0], nullptr);

      uint8_t fillPattern = 0xAF;
      ASSERT_SUCCESS(urEnqueueUSMFill(queues[qIdx], ptrs[0], 1, &fillPattern,
                                      allocSize, 1, &allocEvents[0],
                                      &fillEvents[0]));
      ASSERT_NE(fillEvents[0], nullptr);

      ASSERT_SUCCESS(urEnqueueUSMFreeExp(queues[qIdx], nullptr, ptrs[0], 1,
                                         &fillEvents[0], &freeEvents[0]));
      ASSERT_NE(freeEvents[0], nullptr);

      for (size_t i = 1; i < iterations; i++) {
        ASSERT_SUCCESS(enqueueUSMAllocFunc(queues[qIdx], nullptr, allocSize,
                                           nullptr, 1, &freeEvents[i - 1],
                                           &ptrs[i], &allocEvents[i]));
        ASSERT_NE(ptrs[i], nullptr);
        ASSERT_NE(allocEvents[i], nullptr);

        // Access the memory.
        ASSERT_SUCCESS(urEnqueueUSMFill(queues[qIdx], ptrs[i], 1, &fillPattern,
                                        allocSize, 1, &allocEvents[i],
                                        &fillEvents[i]));
        ASSERT_NE(fillEvents[i], nullptr);

        ASSERT_SUCCESS(urEnqueueUSMFreeExp(queues[qIdx], nullptr, ptrs[i], 1,
                                           &fillEvents[i], &freeEvents[i]));
        ASSERT_NE(freeEvents[i], nullptr);
      }

      ASSERT_SUCCESS(urQueueFinish(queues[qIdx]));

      for (size_t i = 0; i < iterations; i++) {
        ASSERT_SUCCESS(urEventRelease(allocEvents[i]));
        ASSERT_SUCCESS(urEventRelease(fillEvents[i]));
        ASSERT_SUCCESS(urEventRelease(freeEvents[i]));
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }
}

TEST_P(urL0EnqueueAllocMultiQueueMultiDeviceTest,
       SuccessDependantSharedAcrossQueues) {
  const size_t allocSize = std::get<1>(this->GetParam()).allocSize;
  const size_t iterations = std::get<1>(this->GetParam()).iterations;
  const auto enqueueUSMAllocFunc =
      std::get<1>(this->GetParam()).funcParams.enqueueUSMAllocFunc;
  const auto checkUSMSupportFunc =
      std::get<1>(this->GetParam()).funcParams.checkUSMSupportFunc;

  ur_device_usm_access_capability_flags_t USMSupport = 0;
  for (auto &device : devices) {
    ASSERT_SUCCESS(checkUSMSupportFunc(device, USMSupport));
    if (!(USMSupport & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
      GTEST_SKIP() << "Selected USM type is not supported.";
    }
  }

  size_t qIdx = 0;
  std::vector<void *> ptrs(iterations);
  std::vector<ur_event_handle_t> allocEvents(iterations);
  std::vector<ur_event_handle_t> fillEvents(iterations);
  std::vector<ur_event_handle_t> freeEvents(iterations);

  ASSERT_SUCCESS(enqueueUSMAllocFunc(queues[qIdx], nullptr, allocSize, nullptr,
                                     0, nullptr, &ptrs[0], &allocEvents[0]));
  ASSERT_NE(ptrs[0], nullptr);
  ASSERT_NE(allocEvents[0], nullptr);

  uint8_t fillPattern = 0xAF;
  ASSERT_SUCCESS(urEnqueueUSMFill(queues[qIdx], ptrs[0], 1, &fillPattern,
                                  allocSize, 1, &allocEvents[0],
                                  &fillEvents[0]));
  ASSERT_NE(fillEvents[0], nullptr);

  ASSERT_SUCCESS(urEnqueueUSMFreeExp(queues[qIdx], nullptr, ptrs[0], 1,
                                     &fillEvents[0], &freeEvents[0]));
  ASSERT_NE(freeEvents[0], nullptr);

  for (size_t i = 1; i < iterations; i++) {
    qIdx = (qIdx + 1) % queues.size(); // Change queue.
    // Submit subsequent events to different queues.
    ASSERT_SUCCESS(enqueueUSMAllocFunc(queues[qIdx], nullptr, allocSize,
                                       nullptr, 1, &freeEvents[i - 1], &ptrs[i],
                                       &allocEvents[i]));
    ASSERT_NE(ptrs[i], nullptr);
    ASSERT_NE(allocEvents[i], nullptr);

    qIdx = (qIdx + 1) % queues.size(); // Change queue.
    // Access the memory.
    ASSERT_SUCCESS(urEnqueueUSMFill(queues[qIdx], ptrs[i], 1, &fillPattern,
                                    allocSize, 1, &allocEvents[i],
                                    &fillEvents[i]));
    ASSERT_NE(fillEvents[i], nullptr);

    qIdx = (qIdx + 1) % queues.size(); // Change queue.
    ASSERT_SUCCESS(urEnqueueUSMFreeExp(queues[qIdx], nullptr, ptrs[i], 1,
                                       &fillEvents[i], &freeEvents[i]));
    ASSERT_NE(freeEvents[i], nullptr);
  }

  // Wait for the queue that received the last event.
  ASSERT_SUCCESS(urQueueFinish(queues[qIdx]));

  for (size_t i = 0; i < iterations; i++) {
    ASSERT_SUCCESS(urEventRelease(allocEvents[i]));
    ASSERT_SUCCESS(urEventRelease(fillEvents[i]));
    ASSERT_SUCCESS(urEventRelease(freeEvents[i]));
  }
}

TEST_P(urL0EnqueueAllocMultiQueueMultiDeviceTest,
       SuccessDependantSharedAcrossQueuesSyncValidate) {
  const size_t allocSize = std::get<1>(this->GetParam()).allocSize;
  const size_t iterations = std::get<1>(this->GetParam()).iterations;
  const auto enqueueUSMAllocFunc =
      std::get<1>(this->GetParam()).funcParams.enqueueUSMAllocFunc;
  const auto checkUSMSupportFunc =
      std::get<1>(this->GetParam()).funcParams.checkUSMSupportFunc;

  ur_device_usm_access_capability_flags_t USMSupport = 0;
  for (auto &device : devices) {
    ASSERT_SUCCESS(checkUSMSupportFunc(device, USMSupport));
    if (!(USMSupport & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
      GTEST_SKIP() << "Selected USM type is not supported.";
    }
  }

  size_t qIdx = 0;
  std::vector<void *> ptrs(iterations);
  std::vector<ur_event_handle_t> allocEvents(iterations);
  std::vector<ur_event_handle_t> fillEvents(iterations);

  ASSERT_SUCCESS(enqueueUSMAllocFunc(queues[qIdx], nullptr, allocSize, nullptr,
                                     0, nullptr, &ptrs[0], &allocEvents[0]));
  ASSERT_NE(ptrs[0], nullptr);
  ASSERT_NE(allocEvents[0], nullptr);

  uint8_t fillPattern = 0xAF;
  ASSERT_SUCCESS(urEnqueueUSMFill(queues[qIdx], ptrs[0], 1, &fillPattern,
                                  allocSize, 1, &allocEvents[0],
                                  &fillEvents[0]));
  ASSERT_NE(fillEvents[0], nullptr);

  for (size_t i = 1; i < iterations; i++) {
    qIdx = (qIdx + 1) % queues.size(); // Change queue.
    // Submit subsequent events to different queues.
    ASSERT_SUCCESS(enqueueUSMAllocFunc(queues[qIdx], nullptr, allocSize,
                                       nullptr, 1, &fillEvents[i - 1], &ptrs[i],
                                       &allocEvents[i]));
    ASSERT_NE(ptrs[i], nullptr);
    ASSERT_NE(allocEvents[i], nullptr);

    // Wait for the queue with latest submitted event.
    // Validate previous enqeueued fill. Enqueued alloc should finish only after
    // the previous fill event it depends on was finished.
    ASSERT_SUCCESS(urQueueFinish(queues[qIdx]));
    ValidateFill(ptrs[i - 1], fillPattern, allocSize);

    qIdx = (qIdx + 1) % queues.size(); // Change queue.
    // Access the memory.
    ASSERT_SUCCESS(urEnqueueUSMFill(queues[qIdx], ptrs[i], 1, &fillPattern,
                                    allocSize, 1, &allocEvents[i],
                                    &fillEvents[i]));
    ASSERT_NE(fillEvents[i], nullptr);
  }

  // Wait for the queue that received the last event.
  ASSERT_SUCCESS(urQueueFinish(queues[qIdx]));

  for (size_t i = 0; i < iterations; i++) {
    ASSERT_SUCCESS(urEventRelease(allocEvents[i]));
    ASSERT_SUCCESS(urEventRelease(fillEvents[i]));

    ur_event_handle_t freeEvent = nullptr;
    ASSERT_SUCCESS(urEnqueueUSMFreeExp(queues[0], nullptr, ptrs[i], 0, nullptr,
                                       &freeEvent));
    ASSERT_NE(freeEvent, nullptr);
  }
}
