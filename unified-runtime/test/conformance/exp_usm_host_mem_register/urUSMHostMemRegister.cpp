// Copyright (C) 2026 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/fixtures.h"

struct urUSMHostMemRegisterTest : uur::urQueueTest {
  static constexpr size_t allocSize = (1 << 12); // 4KB
  static constexpr uint8_t testValue = 0x77;
  void *alloc = nullptr;

  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::SetUp());

    UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{}, uur::NativeCPU{},
                         uur::OpenCL{}, uur::LevelZero{});

    alloc = aligned_alloc(allocSize, allocSize);
    ASSERT_NE(alloc, nullptr);
  }

  void TearDown() override {
    if (alloc) {
      free(alloc);
    }
    urQueueTest::TearDown();
  }

  void validateBuffer(const void *buffer, const size_t size,
                      const uint8_t expectedValue) {
    const uint8_t *byteBuffer = static_cast<const uint8_t *>(buffer);
    for (size_t i = 0; i < size; ++i) {
      ASSERT_EQ(byteBuffer[i], expectedValue)
          << "Buffer mismatch at index " << i << ": expected "
          << static_cast<int>(expectedValue) << ", got "
          << static_cast<int>(byteBuffer[i]);
    }
  }
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urUSMHostMemRegisterTest);

TEST_P(urUSMHostMemRegisterTest, Success) {
  ASSERT_SUCCESS(urUSMHostAllocRegisterExp(context, alloc, allocSize, nullptr));

  void *alloc2 = nullptr;
  ASSERT_SUCCESS(urUSMHostAlloc(context, nullptr, nullptr, allocSize, &alloc2));

  memset(alloc, testValue, allocSize);
  memset(alloc2, 0, allocSize);

  ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, true, alloc2, alloc, allocSize, 0,
                                    nullptr, nullptr));

  validateBuffer(alloc2, allocSize, testValue);

  ASSERT_SUCCESS(urUSMHostAllocUnregisterExp(context, alloc));
  ASSERT_SUCCESS(urUSMFree(context, alloc2));
}

TEST_P(urUSMHostMemRegisterTest, InvalidNullHandleContext) {
  ASSERT_EQ(urUSMHostAllocRegisterExp(nullptr, alloc, allocSize, nullptr),
            UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urUSMHostMemRegisterTest, InvalidValueAllocSize) {
  ASSERT_EQ(urUSMHostAllocRegisterExp(context, alloc, 0, nullptr),
            UR_RESULT_ERROR_INVALID_VALUE);
}

TEST_P(urUSMHostMemRegisterTest, InvalidNullPointerHostMem) {
  ASSERT_EQ(urUSMHostAllocRegisterExp(context, nullptr, allocSize, nullptr),
            UR_RESULT_ERROR_INVALID_NULL_POINTER);
}
