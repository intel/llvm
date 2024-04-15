//===----------- threadpool.hpp - Native CPU Threadpool
//--------------------===//
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
#include <cstdlib>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <numeric>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace native_cpu {

using worker_task_t = std::function<void(size_t)>;

namespace detail {

class worker_thread {
public:
  // Initializes state, but does not start the worker thread
  worker_thread() noexcept : m_isRunning(false), m_numTasks(0) {}

  // Creates and launches the worker thread
  inline void start(size_t threadId) {
    std::lock_guard<std::mutex> lock(m_workMutex);
    if (this->is_running()) {
      return;
    }
    m_threadId = threadId;
    m_worker = std::thread([this]() {
      while (true) {
        // pin the thread to the cpu
        std::unique_lock<std::mutex> lock(m_workMutex);
        // Wait until there's work available
        m_startWorkCondition.wait(
            lock, [this]() { return !this->is_running() || !m_tasks.empty(); });
        if (!this->is_running() && m_tasks.empty()) {
          // Can only break if there is no more work to be done
          break;
        }
        // Retrieve a task from the queue
        auto task = m_tasks.front();
        m_tasks.pop();

        // Not modifying internal state anymore, can release the mutex
        lock.unlock();

        // Execute the task
        task(m_threadId);
        --m_numTasks;
      }
    });

    m_isRunning = true;
  }

  inline void schedule(const worker_task_t &task) {
    {
      std::lock_guard<std::mutex> lock(m_workMutex);
      // Add the task to the queue
      m_tasks.push(task);
      ++m_numTasks;
    }
    m_startWorkCondition.notify_one();
  }

  size_t num_pending_tasks() const noexcept {
    // m_numTasks is an atomic counter because we don't want to lock the mutex
    // here, num_pending_tasks is only used for heuristics
    return m_numTasks.load();
  }

  // Waits for all tasks to finish and destroys the worker thread
  inline void stop() {
    {
      // Notify the worker thread to stop executing
      std::lock_guard<std::mutex> lock(m_workMutex);
      m_isRunning = false;
    }
    m_startWorkCondition.notify_all();
    if (m_worker.joinable()) {
      // Wait for the worker thread to finish handling the task queue
      m_worker.join();
    }
  }

  // Checks whether the thread pool is currently running threads
  inline bool is_running() const noexcept { return m_isRunning; }

private:
  // Unique ID identifying the thread in the threadpool
  size_t m_threadId;
  std::thread m_worker;

  std::mutex m_workMutex;

  std::condition_variable m_startWorkCondition;

  bool m_isRunning;

  std::queue<worker_task_t> m_tasks;

  std::atomic<size_t> m_numTasks;
};

// Implementation of a thread pool. The worker threads are created and
// ready at construction. This class mainly holds the interface for
// scheduling a task to the most appropriate thread and handling input
// parameters and futures.
class simple_thread_pool {
public:
  simple_thread_pool(size_t numThreads = 0) noexcept : m_isRunning(false) {
    this->resize(numThreads);
    this->start();
  }

  ~simple_thread_pool() { this->stop(); }

  // Creates and launches the worker threads
  inline void start() {
    if (this->is_running()) {
      return;
    }
    size_t threadId = 0;
    for (auto &t : m_workers) {
      t.start(threadId);
      threadId++;
    }
    m_isRunning.store(true, std::memory_order_release);
  }

  // Waits for all tasks to finish and destroys the worker threads
  inline void stop() {
    for (auto &t : m_workers) {
      t.stop();
    }
    m_isRunning.store(false, std::memory_order_release);
  }

  inline void resize(size_t numThreads) {
    char *envVar = std::getenv("SYCL_NATIVE_CPU_HOST_THREADS");
    if (envVar) {
      numThreads = std::stoul(envVar);
    }
    if (numThreads == 0) {
      numThreads = std::thread::hardware_concurrency();
    }
    if (!this->is_running() && (numThreads != this->num_threads())) {
      m_workers = decltype(m_workers)(numThreads);
    }
  }

  inline void schedule(const worker_task_t &task) {
    // Schedule the task on the best available worker thread
    this->best_worker().schedule(task);
  }

  inline bool is_running() const noexcept {
    return m_isRunning.load(std::memory_order_acquire);
  }

  inline size_t num_threads() const noexcept { return m_workers.size(); }

  inline size_t num_pending_tasks() const noexcept {
    return std::accumulate(std::begin(m_workers), std::end(m_workers),
                           size_t(0),
                           [](size_t numTasks, const worker_thread &t) {
                             return (numTasks + t.num_pending_tasks());
                           });
  }

  void wait_for_all_pending_tasks() {
    while (num_pending_tasks() > 0) {
      std::this_thread::yield();
    }
  }

protected:
  // Determines which thread is the most appropriate for having work
  // scheduled
  worker_thread &best_worker() noexcept {
    return *std::min_element(
        std::begin(m_workers), std::end(m_workers),
        [](const worker_thread &w1, const worker_thread &w2) {
          // Prefer threads whose task queues are shorter
          // This is just an approximation, it doesn't need to be exact
          return (w1.num_pending_tasks() < w2.num_pending_tasks());
        });
  }

private:
  std::vector<worker_thread> m_workers;

  std::atomic<bool> m_isRunning;
};
} // namespace detail

template <typename ThreadPoolT> class threadpool_interface {
  ThreadPoolT threadpool;

public:
  void start() { threadpool.start(); }

  void stop() { threadpool.stop(); }

  size_t num_threads() const noexcept { return threadpool.num_threads(); }

  threadpool_interface(size_t numThreads) : threadpool(numThreads) {}
  threadpool_interface() : threadpool(0) {}

  auto schedule_task(worker_task_t &&task) {
    auto workerTask = std::make_shared<std::packaged_task<void(size_t)>>(
        [task](auto &&PH1) { return task(std::forward<decltype(PH1)>(PH1)); });
    threadpool.schedule([=](size_t threadId) { (*workerTask)(threadId); });
    return workerTask->get_future();
  }
};

using threadpool_t = threadpool_interface<detail::simple_thread_pool>;

} // namespace native_cpu
