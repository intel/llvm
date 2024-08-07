//===-- thread_pool.hpp - Simple thread pool --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <sycl/detail/defines.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

class ThreadPool {
  std::vector<std::thread> MLaunchedThreads;

  size_t MThreadCount;
  std::queue<std::function<void()>> MJobQueue;
  std::mutex MJobQueueMutex;
  std::condition_variable MDoSmthOrStop;
  std::atomic_bool MStop;
  std::atomic_uint MJobsInPool;

  void worker() {
    GlobalHandler::instance().registerSchedulerUsage(/*ModifyCounter*/ false);
    std::unique_lock<std::mutex> Lock(MJobQueueMutex);
    while (true) {
      MDoSmthOrStop.wait(
          Lock, [this]() { return !MJobQueue.empty() || MStop.load(); });

      if (MStop.load())
        break;

      std::function<void()> Job = std::move(MJobQueue.front());
      MJobQueue.pop();
      Lock.unlock();

      Job();

      Lock.lock();

      MJobsInPool--;
    }
  }

  void start() {
    MLaunchedThreads.reserve(MThreadCount);

    MStop.store(false);
    MJobsInPool.store(0);

    for (size_t Idx = 0; Idx < MThreadCount; ++Idx)
      MLaunchedThreads.emplace_back([this] { worker(); });
  }

public:
  void drain() {
    while (MJobsInPool != 0)
      std::this_thread::yield();
  }

  ThreadPool(unsigned int ThreadCount = 1) : MThreadCount(ThreadCount) {
    start();
  }

  ~ThreadPool() {
    try {
      finishAndWait();
    } catch (std::exception &e) {
      __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~ThreadPool", e);
    }
  }

  void finishAndWait() {
    MStop.store(true);

    MDoSmthOrStop.notify_all();

    for (std::thread &Thread : MLaunchedThreads)
      if (Thread.joinable())
        Thread.join();
  }

  template <typename T> void submit(T &&Func) {
    {
      std::lock_guard<std::mutex> Lock(MJobQueueMutex);
      MJobQueue.emplace([F = std::move(Func)]() { F(); });
    }
    MJobsInPool++;
    MDoSmthOrStop.notify_one();
  }

  void submit(std::function<void()> &&Func) {
    {
      std::lock_guard<std::mutex> Lock(MJobQueueMutex);
      MJobQueue.emplace(Func);
    }
    MJobsInPool++;
    MDoSmthOrStop.notify_one();
  }
};

} // namespace detail
} // namespace _V1
} // namespace sycl
