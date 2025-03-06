// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.hpp"

#include <iostream>
#include <thread>
#include <vector>

INSTANTIATE_TEST_SUITE_P(
    threadCountForValDeviceTest, valDeviceTestMultithreaded,
    ::testing::Values(2, 8, std::thread::hardware_concurrency()));

TEST_P(valDeviceTestMultithreaded, testUrContextRetainLeakMt) {
  ur_context_handle_t context = nullptr;
  ASSERT_EQ(urContextCreate(1, &device, nullptr, &context), UR_RESULT_SUCCESS);

  std::vector<std::thread> threads;
  for (int i = 0; i < threadCount; i++) {
    threads.emplace_back([&context]() {
      ASSERT_EQ(urContextRetain(context), UR_RESULT_SUCCESS);
      ASSERT_NE(nullptr, context);
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }
}

TEST_P(valDeviceTestMultithreaded, testUrContextReleaseLeakMt) {
  ur_context_handle_t context = nullptr;
  ASSERT_EQ(urContextCreate(1, &device, nullptr, &context), UR_RESULT_SUCCESS);

  std::vector<std::thread> threads;
  for (int i = 0; i < threadCount; i++) {
    threads.emplace_back([&context]() {
      ASSERT_EQ(urContextRelease(context), UR_RESULT_SUCCESS);
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }
}

TEST_P(valDeviceTestMultithreaded, testUrContextRetainReleaseLeakMt) {
  ur_context_handle_t context = nullptr;
  ASSERT_EQ(urContextCreate(1, &device, nullptr, &context), UR_RESULT_SUCCESS);

  std::vector<std::thread> threads;
  for (int i = 0; i < threadCount; i++) {
    threads.emplace_back([&context]() {
      ASSERT_EQ(urContextRetain(context), UR_RESULT_SUCCESS);
      ASSERT_NE(nullptr, context);
      ASSERT_EQ(urContextRelease(context), UR_RESULT_SUCCESS);
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }
}
