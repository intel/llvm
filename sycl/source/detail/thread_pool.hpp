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
  bool MStop = false;
  std::atomic_uint MJobsInPool;

#ifdef _WIN32
  class ThreadExitTracker {
  public:
    void wait(size_t ThreadCount) {
      std::unique_lock<std::mutex> lk(MWorkerExitMutex);
      MWorkerExitCV.wait(
          lk, [&ThreadCount, this] { return MWorkerExitCount == ThreadCount; });
    }

    void signalAboutExit() {
      {
        std::lock_guard<std::mutex> lk(MWorkerExitMutex);
        MWorkerExitCount++;
      }
      MWorkerExitCV.notify_one();
    }

  private:
    std::mutex MWorkerExitMutex;
    std::condition_variable MWorkerExitCV;
    size_t MWorkerExitCount{};
  } WinThreadExitTracker;
#endif

  void worker() {
    GlobalHandler::instance().registerSchedulerUsage(/*ModifyCounter*/ false);
    std::unique_lock<std::mutex> Lock(MJobQueueMutex);
    while (true) {
      MDoSmthOrStop.wait(Lock,
                         [this]() { return !MJobQueue.empty() || MStop; });

      if (MStop) {
#ifdef _WIN32
        WinThreadExitTracker.signalAboutExit();
#endif
        return;
      }

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
#ifndef _WIN32
      finishAndWait(true);
#endif
    } catch (std::exception &e) {
      __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~ThreadPool", e);
    }
  }

  void finishAndWait(bool CanJoinThreads) {
    {
      std::lock_guard<std::mutex> Lock(MJobQueueMutex);
      MStop = true;
    }

    MDoSmthOrStop.notify_all();

#ifdef _WIN32
    if (!CanJoinThreads) {
      WinThreadExitTracker.wait(MThreadCount);
      for (std::thread &Thread : MLaunchedThreads)
        Thread.detach();
      return;
    }
#else
    // We always can join on Linux.
    std::ignore = CanJoinThreads;
#endif

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
