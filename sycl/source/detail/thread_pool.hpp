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
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

class ThreadPool {
  std::vector<std::thread> MLaunchedThreads;

  size_t MThreadCount;
  std::queue<std::function<void()>> MJobQueue;
  std::mutex MJobQueueMutex;
  std::condition_variable MDoSmthOrStop;
  std::atomic_bool MStop;

  // Drain related staff
  std::map<std::thread::id, std::atomic_bool /*needToDrain*/> MDrainSchedule;
  std::atomic_bool MDrain{false};

  void worker() {
    GlobalHandler::instance().registerSchedulerUsage(false);
    std::unique_lock<std::mutex> Lock(MJobQueueMutex);
    std::thread::id ThisThreadId = std::this_thread::get_id();
    while (true) {
      MDoSmthOrStop.wait(Lock, [this, &ThisThreadId]() {
        return !MJobQueue.empty() || MStop.load() ||
               (MDrain.load() && MDrainSchedule[ThisThreadId].load());
      });

      // lets complete enqueued tasks first
      if (MStop.load() && MJobQueue.empty())
        break;

      if (MJobQueue.empty() && MDrain.load() &&
          MDrainSchedule[ThisThreadId].load()) {
        assert(MDrainSchedule[ThisThreadId].exchange(false));
        continue;
      }

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
  void drain() {
    std::cout << "drain begin" << std::endl;
    for (const auto &thread : MLaunchedThreads) {
      MDrainSchedule[thread.get_id()].store(true);
    }
    MDrain.store(true);
    MDoSmthOrStop.notify_all();
    while (std::any_of(MDrainSchedule.cbegin(), MDrainSchedule.cend(),
                       [](const auto &state) { return state.second.load(); }))
      std::this_thread::yield();

    MDrain.store(false);
    std::cout << "drain end" << std::endl;
  }

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
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
