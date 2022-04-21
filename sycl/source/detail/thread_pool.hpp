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

#include <CL/sycl/detail/defines.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

class ThreadPool {
  std::vector<std::thread> MLaunchedThreads;

  size_t MThreadCount;
  std::queue<std::function<void()>> MJobQueue;
  std::mutex MJobQueueMutex;
  std::condition_variable MDoSmthOrStop;
  std::atomic_bool MStop;

  void worker() {
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
    }
  }

  void start() {
    MLaunchedThreads.reserve(MThreadCount);

    MStop.store(false);

    for (size_t Idx = 0; Idx < MThreadCount; ++Idx)
      MLaunchedThreads.emplace_back([this] { worker(); });
  }

public:
  ThreadPool(unsigned int ThreadCount = 1) : MThreadCount(ThreadCount) {
    start();
  }

  ~ThreadPool() { finishAndWait(); }

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

    MDoSmthOrStop.notify_one();
  }

  void submit(std::function<void()> &&Func) {
    {
      std::lock_guard<std::mutex> Lock(MJobQueueMutex);
      MJobQueue.emplace(Func);
    }

    MDoSmthOrStop.notify_one();
  }
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
