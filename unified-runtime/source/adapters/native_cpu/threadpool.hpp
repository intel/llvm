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
#include <forward_list>
#include <future>
#include <iterator>
#include <mutex>
#include <numeric>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace native_cpu {

using worker_task_t = std::packaged_task<void(size_t)>;

namespace detail {

class worker_thread {
public:
  // Initializes state, but does not start the worker thread
  worker_thread(size_t threadId) noexcept
      : m_threadId(threadId), m_isRunning(false), m_numTasks(0) {
    std::lock_guard<std::mutex> lock(m_workMutex);
    if (this->is_running()) {
      return;
    }
    m_worker = std::thread([this]() {
      while (true) {
        std::unique_lock<std::mutex> lock(m_workMutex);
        // Wait until there's work available
        m_startWorkCondition.wait(
            lock, [this]() { return !this->is_running() || !m_tasks.empty(); });
        if (!this->is_running() && m_tasks.empty()) {
          // Can only break if there is no more work to be done
          break;
        }
        // Retrieve a task from the queue
        worker_task_t task = std::move(m_tasks.front());
        m_tasks.pop();

        // Not modifying internal state anymore, can release the mutex
        lock.unlock();

        // Execute the task
        task(m_threadId);
        --m_numTasks;
      }
    });

    m_isRunning.store(true, std::memory_order_release);
  }

  inline void schedule(worker_task_t &&task) {
    {
      std::lock_guard<std::mutex> lock(m_workMutex);
      // Add the task to the queue
      m_tasks.emplace(std::move(task));
      ++m_numTasks;
    }
    m_startWorkCondition.notify_one();
  }

  size_t num_pending_tasks() const noexcept {
    // m_numTasks is an atomic counter because we don't want to lock the mutex
    // here, num_pending_tasks is only used for heuristics
    return m_numTasks.load(std::memory_order_acquire);
  }

  // Waits for all tasks to finish and destroys the worker thread
  inline void stop() {
    {
      std::lock_guard<std::mutex> lock(m_workMutex);
      m_isRunning.store(false, std::memory_order_release);
      m_startWorkCondition.notify_all();
    }
    if (m_worker.joinable()) {
      // Wait for the worker thread to finish handling the task queue
      m_worker.join();
    }
  }

  // Checks whether the thread pool is currently running threads
  inline bool is_running() const noexcept {
    return m_isRunning.load(std::memory_order_acquire);
  }

private:
  // Unique ID identifying the thread in the threadpool
  const size_t m_threadId;

  std::thread m_worker;

  std::mutex m_workMutex;

  std::condition_variable m_startWorkCondition;

  std::atomic<bool> m_isRunning;

  std::queue<worker_task_t> m_tasks;

  std::atomic<size_t> m_numTasks;
};

// Implementation of a thread pool. The worker threads are created and
// ready at construction. This class mainly holds the interface for
// scheduling a task to the most appropriate thread and handling input
// parameters and futures.
class simple_thread_pool {
public:
  simple_thread_pool() noexcept
      : m_isRunning(false), m_numThreads(get_num_threads()) {
    for (size_t i = 0; i < m_numThreads; i++) {
      m_workers.emplace_front(i);
    }
    m_isRunning.store(true, std::memory_order_release);
  }

  ~simple_thread_pool() {
    for (auto &t : m_workers) {
      t.stop();
    }
    m_isRunning.store(false, std::memory_order_release);
  }

  inline void schedule(worker_task_t &&task) {
    // Schedule the task on the best available worker thread
    this->best_worker().schedule(std::move(task));
  }

  inline bool is_running() const noexcept {
    return m_isRunning.load(std::memory_order_acquire);
  }

  inline size_t num_threads() const noexcept { return m_numThreads; }

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
  static size_t get_num_threads() {
    size_t numThreads;
    char *envVar = std::getenv("SYCL_NATIVE_CPU_HOST_THREADS");
    if (envVar) {
      numThreads = std::stoul(envVar);
    } else {
      numThreads = std::thread::hardware_concurrency();
    }
    return numThreads;
  }

  std::forward_list<worker_thread> m_workers;

  std::atomic<bool> m_isRunning;

  const size_t m_numThreads;
};
} // namespace detail

template <typename ThreadPoolT> class threadpool_interface {
  ThreadPoolT threadpool;

public:
  size_t num_threads() const noexcept { return threadpool.num_threads(); }

  threadpool_interface() : threadpool() {}

  template <class T> std::future<void> schedule_task(T &&task) {
    auto workerTask = std::packaged_task<void(size_t)>(std::forward<T>(task));
    auto ret = workerTask.get_future();
    threadpool.schedule(std::move(workerTask));
    return ret;
  }
};
using simple_threadpool_t = threadpool_interface<detail::simple_thread_pool>;

class TasksInfo_TP {
  using FType = std::future<void>;
  std::vector<FType> futures;

public:
  void schedule(FType &&f) { futures.emplace_back(std::move(f)); }
  void wait_all() {
    for (auto &f : futures)
      f.wait();
  }
  TasksInfo_TP(simple_threadpool_t &) {}
};

template <class TP, class TaskInfo> struct Scheduler_base {
  TP &ref;
  TaskInfo ti;
  Scheduler_base(TP &ref_) : ref(ref_), ti(ref_) {}
  TaskInfo getMovedTaskInfo() { return std::move(ti); }
  static constexpr bool CanWaitInThread() { return true; }
};

template <class TP> struct Scheduler : Scheduler_base<TP, TasksInfo_TP> {
  using Scheduler_base<TP, TasksInfo_TP>::Scheduler_base;

  template <class T> void schedule(T &&task) {
    this->ti.schedule(this->ref.schedule_task(std::forward<T>(task)));
  }
};

template <class TPType> inline Scheduler<TPType> getScheduler(TPType &tp) {
  return Scheduler<TPType>(tp);
}

} // namespace native_cpu

#ifdef NATIVECPU_WITH_ONETBB
// Simple TBB backend
#include "oneapi/tbb.h"
namespace native_cpu {

class TBB_threadpool {
  oneapi::tbb::task_group tasks;

public:
  void wait_all() { tasks.wait(); }
  oneapi::tbb::task_group &Tasks() { return tasks; }
  size_t num_threads() const noexcept {
    return oneapi::tbb::info::default_concurrency();
  }
};

class TBB_TasksInfo {
  TBB_threadpool *tp;

public:
  void wait_all() { tp->wait_all(); }
  TBB_TasksInfo(TBB_threadpool &t) : tp(&t) {}
};

template <>
struct Scheduler<TBB_threadpool>
    : Scheduler_base<TBB_threadpool, TBB_TasksInfo> {
  using Scheduler_base<TBB_threadpool, TBB_TasksInfo>::Scheduler_base;
  template <class T> void schedule(T &&task_) {
    ref.Tasks().run([task = std::move(task_)]() {
      auto thread_id = tbb::this_task_arena::current_thread_index();
      assert(thread_id >= 0 &&
             thread_id < oneapi::tbb::info::default_concurrency());
      task(thread_id);
    });
  }
  static constexpr bool CanWaitInThread() { return false; }
};

using tasksinfo_t = TBB_TasksInfo;
using threadpool_t = TBB_threadpool;
} // namespace native_cpu

#else
// The default backend
namespace native_cpu {
using tasksinfo_t = TasksInfo_TP;
using threadpool_t = simple_threadpool_t;
} // namespace native_cpu
#endif
