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
  size_t numQueues;
  size_t iterations;
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

struct urL0EnqueueAllocMultiQueueTest
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
  ASSERT_SUCCESS(
      urEnqueueUSMDeviceAllocExp(queue, nullptr, ARRAY_SIZE * sizeof(uint32_t),
                                 nullptr, 0, nullptr, &ptr2, nullptr));
  ASSERT_NE(ptr, nullptr);

  ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, ptr2));
  ASSERT_SUCCESS(urKernelSetArgValue(kernel, 1, sizeof(DATA), nullptr, &DATA));
  Launch1DRange(ARRAY_SIZE);

  ValidateEnqueueFree(ptr2);
}

UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urL0EnqueueAllocMultiQueueTest,
    ::testing::ValuesIn({
        EnqueueAllocMultiQueueTestParam{1024, 8, 256, urEnqueueUSMHostAllocExp,
                                        uur::GetDeviceUSMHostSupport},
        EnqueueAllocMultiQueueTestParam{1024, 8, 256,
                                        urEnqueueUSMSharedAllocExp,
                                        uur::GetDeviceUSMSingleSharedSupport},
        EnqueueAllocMultiQueueTestParam{1024, 8, 256,
                                        urEnqueueUSMDeviceAllocExp,
                                        uur::GetDeviceUSMDeviceSupport},
    }),
    uur::deviceTestWithParamPrinter<EnqueueAllocMultiQueueTestParam>);

TEST_P(urL0EnqueueAllocMultiQueueTest, Success) {
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

  std::mutex mtx;
  std::condition_variable cv1;
  size_t cvCounter1 = numQueues;
  std::condition_variable cv2;
  size_t cvCounter2 = numQueues;

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

      // Synchronize threads.
      {
        std::unique_lock<std::mutex> lock(mtx);
        cvCounter1--;
        if (cvCounter1 > 0) {
          cv1.wait(lock, [&] { return cvCounter1 == 0; });
        } else {
          cv1.notify_all();
        }
      }

      uint8_t fillPattern = 0xAF;
      for (size_t i = 0; i < iterations; i++) {
        ASSERT_SUCCESS(urEnqueueUSMFill(queues[idx], ptrs[i], 1, &fillPattern,
                                        allocSize, 0, nullptr, nullptr));
      }

      ASSERT_SUCCESS(urQueueFinish(queues[idx]));

      // Synchronize threads.
      {
        std::unique_lock<std::mutex> lock(mtx);
        cvCounter2--;
        if (cvCounter2 > 0) {
          cv2.wait(lock, [&] { return cvCounter2 == 0; });
        } else {
          cv2.notify_all();
        }
      }

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

TEST_P(urL0EnqueueAllocMultiQueueTest, SuccessEventDependant) {
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

      uint8_t fillPattern = 0xAF;
      for (size_t i = 0; i < iterations; i++) {
        ur_event_handle_t allocEvent = nullptr;
        ASSERT_SUCCESS(enqueueUSMAllocFunc(queues[idx], nullptr, allocSize,
                                           nullptr, 0, nullptr, &ptrs[i],
                                           &allocEvent));
        ASSERT_NE(ptrs[i], nullptr);
        ASSERT_NE(allocEvent, nullptr);

        ur_event_handle_t fillEvent = nullptr;
        // Access the memory.
        ASSERT_SUCCESS(urEnqueueUSMFill(queues[idx], ptrs[i], 1, &fillPattern,
                                        allocSize, 1, &allocEvent, &fillEvent));
        ASSERT_NE(fillEvent, nullptr);

        ur_event_handle_t freeEvent = nullptr;
        ASSERT_SUCCESS(urEnqueueUSMFreeExp(queues[idx], nullptr, ptrs[i], 1,
                                           &fillEvent, &freeEvent));
        ASSERT_NE(freeEvent, nullptr);

        ASSERT_SUCCESS(urQueueFinish(queues[idx]));

        urEventRelease(allocEvent);
        urEventRelease(fillEvent);
        urEventRelease(freeEvent);
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }
}
