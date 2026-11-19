// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %maybe-v1 ./cross_queue_wait-test
// RUN: %maybe-v2 ./cross_queue_wait-test

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <thread>
#include <uur/fixtures.h>
#include <vector>

using namespace std::chrono_literals;

// Several threads enqueue commands on QueueA that wait on the same event from
// QueueB while another thread keeps calling urQueueFinish(QueueA). The
// implementation must keep a consistent queue -> event lock order on all of
// these paths, otherwise they can deadlock.
struct urLevelZeroCrossQueueWaitFinishTest : uur::urContextTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urContextTest::SetUp());
    ur_queue_properties_t Props = {UR_STRUCTURE_TYPE_QUEUE_PROPERTIES, nullptr,
                                   UR_QUEUE_FLAG_SUBMISSION_BATCHED};
    ASSERT_SUCCESS(urQueueCreate(context, device, &Props, &QueueA));
    ASSERT_SUCCESS(urQueueCreate(context, device, &Props, &QueueB));
    ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                    sizeof(uint32_t), &Mem));
  }

  void TearDown() override {
    if (Mem) {
      EXPECT_SUCCESS(urUSMFree(context, Mem));
    }
    if (QueueB) {
      EXPECT_SUCCESS(urQueueRelease(QueueB));
    }
    if (QueueA) {
      EXPECT_SUCCESS(urQueueRelease(QueueA));
    }
    UUR_RETURN_ON_FATAL_FAILURE(urContextTest::TearDown());
  }

  ur_queue_handle_t QueueA = nullptr;
  ur_queue_handle_t QueueB = nullptr;
  void *Mem = nullptr;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urLevelZeroCrossQueueWaitFinishTest);

TEST_P(urLevelZeroCrossQueueWaitFinishTest, NoDeadlock) {
  constexpr uint32_t Rounds = 5000;
  constexpr int NumEnqueuers = 4;
  constexpr int WaitersPerEvent = 4;

  std::atomic<bool> Stop{false};
  std::atomic<ur_result_t> Result{UR_RESULT_SUCCESS};
  std::atomic<ur_event_handle_t> SharedEvent{nullptr};
  std::atomic<uint32_t> Round{0};
  std::atomic<int> Done{0};

  auto Fail = [&](ur_result_t Res) {
    Result.store(Res);
    Stop.store(true);
  };

  std::vector<std::future<void>> Threads;

  // Each round, produce one not-yet-completed event on QueueB.
  Threads.push_back(std::async(std::launch::async, [&] {
    for (uint32_t R = 1; R <= Rounds && !Stop.load(); ++R) {
      ur_event_handle_t Event = nullptr;
      ur_result_t Res = urEnqueueUSMFill(QueueB, Mem, sizeof(R), &R, sizeof(R),
                                         0, nullptr, &Event);
      if (Res != UR_RESULT_SUCCESS) {
        Fail(Res);
        break;
      }
      SharedEvent.store(Event);
      Done.store(0);
      Round.store(R);
      while (Done.load() < NumEnqueuers && !Stop.load())
        std::this_thread::yield();
      Res = urEventRelease(Event);
      if (Res != UR_RESULT_SUCCESS) {
        Fail(Res);
        break;
      }
    }
    Stop.store(true);
  }));

  // Make several QueueA commands depend on that event.
  for (int T = 0; T < NumEnqueuers; ++T) {
    Threads.push_back(std::async(std::launch::async, [&] {
      uint32_t Seen = 0;
      while (true) {
        while (Round.load() == Seen && !Stop.load())
          std::this_thread::yield();
        if (Stop.load())
          break;
        Seen = Round.load();
        ur_event_handle_t Event = SharedEvent.load();
        for (int J = 0; J < WaitersPerEvent; ++J) {
          ur_result_t Res = urEnqueueEventsWait(QueueA, 1, &Event, nullptr);
          if (Res != UR_RESULT_SUCCESS)
            Fail(Res);
        }
        Done.fetch_add(1);
      }
    }));
  }

  // Concurrently clean up completed QueueA commands.
  Threads.push_back(std::async(std::launch::async, [&] {
    while (!Stop.load()) {
      ur_result_t Res = urQueueFinish(QueueA);
      if (Res != UR_RESULT_SUCCESS)
        Fail(Res);
    }
  }));

  // A deadlock leaves the threads blocked forever, so there is nothing to
  // join; report it and exit instead of hanging the test run.
  auto Deadline = std::chrono::steady_clock::now() + 120s;
  for (auto &Thread : Threads) {
    if (Thread.wait_until(Deadline) != std::future_status::ready) {
      std::fprintf(stderr, "Deadlock between cross-queue wait and "
                           "urQueueFinish\n");
      std::_Exit(EXIT_FAILURE);
    }
  }

  // Drain both queues before checking the result so TearDown never frees Mem
  // while commands using it are still in flight.
  ASSERT_SUCCESS(urQueueFinish(QueueB));
  ASSERT_SUCCESS(urQueueFinish(QueueA));
  ASSERT_SUCCESS(Result.load());
}
