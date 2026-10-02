// REQUIRES: level_zero_v2_adapter
// REQUIRES: aspect-ext_oneapi_async_memory_alloc
// REQUIRES: aspect-usm_host_allocations
// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// tests-10-02 E14: Async malloc/free and pools follow captured original work
// - queue helpers inferring the last event.
//
// On an in-order queue, the async_malloc and async_malloc_from_pool queue
// helpers order the allocation after the queue's last command. Kernel K,
// held behind a host task gate, writes an allocation P and returns E; E is
// re-signaled on another queue and that signal completes. A helper allocation
// on the same queue must still come after K, and so must the kernel K2 that
// follows it and copies P.
//
// The helpers order the allocation through the queue's captured last signal,
// not through ext_oneapi_get_last_event, so review-10-02 #7 does not apply to
// them; the last-event query itself is covered by last_event_resignal.cpp.
// The "still pending" check is best effort.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/memory_pool.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace syclex = sycl::ext::oneapi::experimental;
using namespace std::chrono_literals;

constexpr size_t N = 1024;

// A gate an ordinary host task blocks on. The wait is bounded, so a test bug
// fails instead of hanging the suite.
struct Gate {
  std::mutex M;
  std::condition_variable CV;
  bool IsOpen = false;
  std::atomic<bool> TimedOut{false};

  void wait() {
    std::unique_lock<std::mutex> Lock(M);
    if (!CV.wait_for(Lock, 60s, [&] { return IsOpen; }))
      TimedOut = true;
  }
  void open() {
    {
      std::lock_guard<std::mutex> Lock(M);
      IsOpen = true;
    }
    CV.notify_all();
  }
};

// Opens the gate on every exit path.
struct GateOpener {
  std::shared_ptr<Gate> G;
  ~GateOpener() { G->open(); }
};

static int Failures = 0;

static void check(bool Cond, const std::string &Case, const char *What) {
  if (!Cond) {
    std::cerr << "FAILED (" << Case << "): " << What << std::endl;
    ++Failures;
  }
}

static bool isComplete(const sycl::event &E) {
  return E.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete;
}

// FromPool selects async_malloc_from_pool, otherwise async_malloc is used.
static void runCase(sycl::queue &Q1, sycl::queue &Q2, syclex::memory_pool &Pool,
                    bool FromPool) {
  const std::string Case =
      FromPool ? "async_malloc_from_pool(queue)" : "async_malloc(queue)";
  int *P = nullptr;
  Q1.submit([&](sycl::handler &CGH) {
    P = static_cast<int *>(
        syclex::async_malloc_from_pool(CGH, N * sizeof(int), Pool));
  });
  Q1.fill(P, 0, N);
  Q1.wait();
  int *Out = sycl::malloc_host<int>(N, Q1);
  for (size_t I = 0; I < N; ++I)
    Out[I] = -1;

  auto G = std::make_shared<Gate>();
  GateOpener Opener{G};
  Q1.submit([&](sycl::handler &CGH) { CGH.host_task([G] { G->wait(); }); });
  sycl::event E = Q1.parallel_for(sycl::range<1>(N), [=](sycl::id<1> I) {
    P[I] = static_cast<int>(I[0]) + 1;
  });

  syclex::enqueue_signal_event(Q2, E);
  E.wait();
  check(isComplete(E), Case, "the new signal is not complete");

  int *P2 = static_cast<int *>(
      FromPool ? syclex::async_malloc_from_pool(Q1, N * sizeof(int), Pool)
               : syclex::async_malloc(Q1, sycl::usm::alloc::device,
                                      N * sizeof(int)));
  sycl::event K2 = Q1.parallel_for(sycl::range<1>(N), [=](sycl::id<1> I) {
    P2[I] = P[I];
    Out[I] = P[I];
  });

  // Best effort: give a wrongly released kernel time to run.
  std::this_thread::sleep_for(100ms);
  check(!isComplete(K2), Case,
        "the kernel after the helper allocation did not stay pending");

  G->open();
  Q1.wait();
  check(!G->TimedOut, Case, "the gate timed out");
  bool Ok = true;
  for (size_t I = 0; I < N; ++I)
    Ok &= Out[I] == static_cast<int>(I) + 1;
  check(Ok, Case, "the kernel after the helper allocation ran too early");

  syclex::async_free(Q1, P2);
  syclex::async_free(Q1, P);
  Q1.wait();
  sycl::free(Out, Q1);
}

int main() {
  sycl::queue Q1{sycl::property::queue::in_order{}};
  sycl::context Ctx = Q1.get_context();
  sycl::device Dev = Q1.get_device();
  sycl::queue Q2{Ctx, Dev, sycl::property::queue::in_order{}};

  {
    syclex::memory_pool Pool(Ctx, Dev, sycl::usm::alloc::device);
    runCase(Q1, Q2, Pool, /*FromPool=*/true);
    runCase(Q1, Q2, Pool, /*FromPool=*/false);
    // Drain every queue before the pool is destroyed.
    Q1.wait();
    Q2.wait();
  }
  return Failures ? 1 : 0;
}
