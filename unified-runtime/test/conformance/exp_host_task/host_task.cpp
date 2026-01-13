// Copyright (C) 2026 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ur_api.h"
#include <gtest/gtest.h>
#include <thread>
#include <uur/fixtures.h>
#include <uur/known_failure.h>
#include <vector>

using T = uint32_t;
using namespace std::chrono_literals;

struct urEnqueueHostTaskTest : uur::urKernelExecutionTest {
  void SetUp() override {
    program_name = "fill";
    UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());

    ur_bool_t host_task_support = false;
    ASSERT_SUCCESS(urDeviceGetInfo(
        device, UR_DEVICE_INFO_ENQUEUE_HOST_TASK_SUPPORT_EXP,
        sizeof(host_task_support), &host_task_support, nullptr));
    if (!host_task_support) {
      GTEST_SKIP();
    }
  }

  uint32_t val = 42;
  size_t global_size = 32;
  size_t global_offset = 0;
  size_t n_dimensions = 1;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEnqueueHostTaskTest);

void IncHostTask(void *data) {
  int *num = (int *)data;
  *num += 1;
}

TEST_P(urEnqueueHostTaskTest, SuccessWithEvents) {
  ur_mem_handle_t buffer = nullptr;
  AddBuffer1DArg(sizeof(val) * global_size, &buffer);
  AddPodArg(val);
  ur_event_handle_t KernelEvent;
  ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                       &global_offset, &global_size, nullptr,
                                       nullptr, 0, nullptr, &KernelEvent));
  int userdata = 1;
  ur_event_handle_t HostTaskEvent;
  urEnqueueHostTaskExp(queue, IncHostTask, &userdata, nullptr, 1, &KernelEvent,
                       &HostTaskEvent);

  urEventWait(1, &HostTaskEvent);
  ASSERT_EQ(userdata, 2);

  ASSERT_SUCCESS(urQueueFinish(queue));
  ValidateBuffer(buffer, sizeof(val) * global_size, val);
}

#define FILL_VAL 42
#define NUM_ELEM 32
#define BUF_SIZE (NUM_ELEM * sizeof(uint32_t))

void MemsetHostTask(void *data) {
  uint32_t buf[NUM_ELEM];
  // B == (42, 42, ...)
  memset(buf, FILL_VAL, BUF_SIZE);
  assert(memcmp(data, buf, BUF_SIZE) == 0);

  // B = (41, 41, ...)
  memset(data, FILL_VAL - 1, BUF_SIZE);
}

TEST_P(urEnqueueHostTaskTest, SuccessCopySandwich) {
  uint32_t bufA[NUM_ELEM];
  memset(bufA, FILL_VAL, BUF_SIZE);

  uint32_t bufB[NUM_ELEM] = {0};

  // A (42, 42, ...) -> B (0, 0, ...)
  ASSERT_SUCCESS(
      urEnqueueUSMMemcpy(queue, 0, bufB, bufA, BUF_SIZE, 0, nullptr, nullptr));

  // B = (41, 41, ...)
  urEnqueueHostTaskExp(queue, MemsetHostTask, &bufB, nullptr, 0, nullptr,
                       nullptr);

  // B (41, 41, ...) -> A (42, 42, ...)
  ASSERT_SUCCESS(
      urEnqueueUSMMemcpy(queue, 0, bufA, bufB, BUF_SIZE, 0, nullptr, nullptr));

  ASSERT_SUCCESS(urQueueFinish(queue));

  uint32_t testBuf[NUM_ELEM];
  memset(testBuf, FILL_VAL - 1, BUF_SIZE);

  ASSERT_EQ(memcmp(testBuf, bufA, BUF_SIZE), 0); // A == (41, 41, ...)
}

#define NTHREADS 8
#define OUTER_ITERS 10
#define INNER_ITERS 10

TEST_P(urEnqueueHostTaskTest, Multithreaded) {
  std::vector<std::thread> threads;
  std::atomic<int> global_counter{0};

  for (int i = 0; i < NTHREADS; ++i) {
    threads.emplace_back([&, i]() {
      int local_counter = 0;

      ur_queue_handle_t queue;
      ur_queue_properties_t queueProperties = {
          UR_STRUCTURE_TYPE_QUEUE_PROPERTIES, nullptr,
          UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE};
      ASSERT_SUCCESS(urQueueCreate(context, device, &queueProperties, &queue));

      for (int iter = 0; iter < OUTER_ITERS; ++iter) {
        for (int task = 0; task < INNER_ITERS; ++task) {
          ur_event_handle_t event = nullptr;
          ASSERT_SUCCESS(urEnqueueHostTaskExp(
              queue,
              [](void *data) {
                std::this_thread::sleep_for(100us);
                int *counter = static_cast<int *>(data);
                ++(*counter);
              },
              &local_counter, nullptr, 0, nullptr, &event));

          ASSERT_NE(event, nullptr);
          urEventRelease(event);
        }
        urQueueFinish(queue);
      }

      global_counter += local_counter;

      urQueueRelease(queue);
    });
  }

  for (auto &th : threads) {
    th.join();
  }

  ASSERT_EQ(global_counter, NTHREADS * OUTER_ITERS * INNER_ITERS);
}
