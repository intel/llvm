// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %use-mock %validate leaks_mt-test | FileCheck %s

#include "fixtures.hpp"

#include <iostream>
#include <thread>
#include <vector>

INSTANTIATE_TEST_SUITE_P(
    threadCountForValDeviceTest, valDeviceTestMultithreaded,
    ::testing::Values(2, 8, std::thread::hardware_concurrency()));

// CHECK: [ RUN      ] threadCountForValDeviceTest/valDeviceTestMultithreaded.testUrContextRetainLeakMt/0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 2
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 3
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[ERROR]: Retained 3 reference(s) to handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[ERROR]: Handle {{[0-9xa-fA-F]+}} was recorded for first time here:
// CHECK: [ RUN      ] threadCountForValDeviceTest/valDeviceTestMultithreaded.testUrContextRetainLeakMt/1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 2
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 3
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 4
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 5
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 6
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 7
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 8
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 9
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[ERROR]: Retained 9 reference(s) to handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[ERROR]: Handle {{[0-9xa-fA-F]+}} was recorded for first time here:

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

// CHECK: [ RUN      ] threadCountForValDeviceTest/valDeviceTestMultithreaded.testUrContextReleaseLeakMt/0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[ERROR]: Attempting to release nonexistent handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to -1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[ERROR]: Retained -1 reference(s) to handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[ERROR]: Handle {{[0-9xa-fA-F]+}} was recorded for first time here:
// CHECK: [ RUN      ] threadCountForValDeviceTest/valDeviceTestMultithreaded.testUrContextReleaseLeakMt/1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[ERROR]: Attempting to release nonexistent handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to -1
// CHECK-NEXT: <VALIDATION>[ERROR]: Attempting to release nonexistent handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to -2
// CHECK-NEXT: <VALIDATION>[ERROR]: Attempting to release nonexistent handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to -3
// CHECK-NEXT: <VALIDATION>[ERROR]: Attempting to release nonexistent handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to -4
// CHECK-NEXT: <VALIDATION>[ERROR]: Attempting to release nonexistent handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to -5
// CHECK-NEXT: <VALIDATION>[ERROR]: Attempting to release nonexistent handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to -6
// CHECK-NEXT: <VALIDATION>[ERROR]: Attempting to release nonexistent handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to -7
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[ERROR]: Retained -7 reference(s) to handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[ERROR]: Handle {{[0-9xa-fA-F]+}} was recorded for first time here:
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

// CHECK: [ RUN      ] threadCountForValDeviceTest/valDeviceTestMultithreaded.testUrContextRetainReleaseLeakMt/0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to {{[1-9]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to {{[1-9]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to {{[1-9]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[ERROR]: Retained 1 reference(s) to handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[ERROR]: Handle {{[0-9xa-fA-F]+}} was recorded for first time here:
// CHECK: [ RUN      ] threadCountForValDeviceTest/valDeviceTestMultithreaded.testUrContextRetainReleaseLeakMt/1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to {{[1-9]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to {{[1-9]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to {{[1-9]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to {{[1-9]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to {{[1-9]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to {{[1-9]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to {{[1-9]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to {{[1-9]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to {{[1-9]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to {{[1-9]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to {{[1-9]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to {{[1-9]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to {{[1-9]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to {{[1-9]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to {{[1-9]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[ERROR]: Retained 1 reference(s) to handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[ERROR]: Handle {{[0-9xa-fA-F]+}} was recorded for first time here:
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
