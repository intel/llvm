// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// Regression test for CMPLRLLVM-77682 / Argonne #157.
//
// With multiple worker threads submitting to an in-order queue, if each
// thread (a) holds an application-level mutex across its q.submit() calls
// and (b) the submitted host_task itself acquires that same mutex, a
// three-way deadlock can arise between:
//   - a submitter blocked on the scheduler graph write lock in
//     Scheduler::addCG,
//   - another submitter that holds the graph read lock inside q.wait() and
//     is blocked in event_impl::waitInternal for a host_task's completion,
//   - the ThreadPool worker running that host_task, blocked acquiring the
//     application mutex held by the first submitter.
//
// The pattern models a Python-GIL-style lock held across SYCL submissions,
// which is how Argonne originally hit this from a PyTorch/XPU workload.

#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>

namespace {

std::mutex g_gil; // stands in for CPython's GIL
constexpr int ITERS = 200;
constexpr int HOST_TASKS_PER_ITER = 2;

void worker(sycl::queue q, int *p) {
  for (int i = 0; i < ITERS; ++i) {
    sycl::event k = q.single_task([=]() { *p = 0; });
    {
      std::lock_guard<std::mutex> OuterLock(g_gil);
      for (int h = 0; h < HOST_TASKS_PER_ITER; ++h) {
        q.submit([&](sycl::handler &cgh) {
          cgh.depends_on(k);
          cgh.host_task([]() { std::lock_guard<std::mutex> InnerLock(g_gil); });
        });
      }
    }
    q.wait();
  }
}

} // namespace

int main() {
  // In-process watchdog: on regression, exit fast with a clear message
  // instead of waiting for lit's global maxIndividualTestTime.
  std::atomic<bool> done{false};
  std::thread([&] {
    for (int i = 0; i < 200; ++i) { // ~20s
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (done.load(std::memory_order_relaxed))
        return;
    }
    std::fprintf(
        stderr,
        "TIMEOUT: in-order host_task + app-mutex deadlock (CMPLRLLVM-77682)\n");
    std::_Exit(2);
  }).detach();

  sycl::queue q{sycl::property::queue::in_order{}};
  int *p = sycl::malloc_device<int>(2, q);

  std::thread t0(worker, q, &p[0]);
  std::thread t1(worker, q, &p[1]);
  t0.join();
  t1.join();

  sycl::free(p, q);
  done.store(true, std::memory_order_relaxed);
  return 0;
}
