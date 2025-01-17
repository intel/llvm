// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "context.hpp"
#include "fixtures.h"
#include "queue.hpp"
#include "uur/raii.h"

using urHipContextTest = uur::urDeviceTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urHipContextTest);

TEST_P(urHipContextTest, ActiveContexts) {
  uur::raii::Context context = nullptr;
  ASSERT_SUCCESS(urContextCreate(1, &device, nullptr, context.ptr()));
  ASSERT_NE(context, nullptr);

  uur::raii::Queue queue = nullptr;
  ASSERT_SUCCESS(urQueueCreate(context, device, nullptr, queue.ptr()));
  ASSERT_NE(queue, nullptr);

  // ensure that the queue has the correct context
  ASSERT_EQ(context, queue->getContext());
}

TEST_P(urHipContextTest, ActiveContextsThreads) {
  uur::raii::Context context1 = nullptr;
  ASSERT_SUCCESS(urContextCreate(1, &device, nullptr, context1.ptr()));
  ASSERT_NE(context1, nullptr);

  uur::raii::Context context2 = nullptr;
  ASSERT_SUCCESS(urContextCreate(1, &device, nullptr, context2.ptr()));
  ASSERT_NE(context2, nullptr);

  // setup synchronization
  std::mutex mtx;
  std::condition_variable cv;
  bool released = false;
  bool thread_done = false;

  auto test_thread = std::thread([&] {
    {
      uur::raii::Queue queue = nullptr;
      ASSERT_SUCCESS(urQueueCreate(context1, device, nullptr, queue.ptr()));
      ASSERT_NE(queue, nullptr);

      // ensure queue has the correct context
      ASSERT_EQ(queue->getContext(), context1);
    }

    // mark the first set of processing as done and notify the main thread
    {
      std::unique_lock<std::mutex> lock(mtx);
      thread_done = true;
    }
    cv.notify_one();

    // wait for main thread to release the first context
    {
      std::unique_lock<std::mutex> lock(mtx);
      cv.wait(lock, [&] { return released; });
    }

    {
      // create a queue with the 2nd context
      uur::raii::Queue queue = nullptr;
      ASSERT_SUCCESS(urQueueCreate(context2, device, nullptr, queue.ptr()));
      ASSERT_NE(queue, nullptr);

      // ensure the queue has the correct context
      ASSERT_EQ(queue->getContext(), context2);
    }
  });

  // wait for the thread to be done with the first queue
  std::unique_lock<std::mutex> lock(mtx);
  cv.wait(lock, [&] { return thread_done; });

  released = true;
  lock.unlock();
  cv.notify_one();

  // wait for thread to finish
  test_thread.join();
}
#pragma GCC diagnostic pop
