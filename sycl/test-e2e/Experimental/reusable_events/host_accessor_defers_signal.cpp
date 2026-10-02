// REQUIRES: level_zero_v2_adapter
// REQUIRES: aspect-usm_device_allocations, aspect-usm_host_allocations

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// tests-10-02 E07: A host accessor defers the signal like a host task.
//
// A buffer kernel is blocked in the SYCL runtime by a live host accessor (and
// a copy of it), and E is signaled behind that kernel on an in-order queue. A
// consumer on another queue captures E. Releasing only one of the accessor
// copies must not release the kernel, the consumer or E; releasing the last
// copy must release all of them. Variants: a host accessor built with its
// constructor and with buffer::get_host_access, each with and without E being
// re-signaled elsewhere (and the new signal completing) before the accessor is
// released. Corresponds to unit scenario U04 (host accessor dependency).

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include <chrono>
#include <cstdlib>
#include <exception>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;
using namespace std::chrono_literals;

constexpr size_t N = 1024;
// Bound for anything that must finish: a missing wakeup fails, not hangs.
constexpr auto Timeout = 30s;
// Observation window of the best-effort "must still be pending" checks.
constexpr auto Settle = 100ms;

static int Failures = 0;

static void check(bool Cond, const char *What) {
  if (!Cond) {
    std::cerr << "FAILED: " << What << std::endl;
    ++Failures;
  }
}

template <typename F>
static void checkAll(const int *Res, F Expected, const char *What) {
  for (size_t I = 0; I < N; ++I) {
    if (Res[I] != Expected(I)) {
      std::cerr << "FAILED: " << What << ": at " << I << " got " << Res[I]
                << ", expected " << Expected(I) << std::endl;
      ++Failures;
      return;
    }
  }
}

// A thread which may still be blocked cannot be joined: end the process.
[[noreturn]] static void fatal(const char *What) {
  std::cerr << "FATAL: " << What << std::endl;
  std::_Exit(1);
}

// Runs Fn on a detached thread; the future is ready when Fn returns.
template <typename F> static std::shared_future<void> startAsync(F Fn) {
  auto Done = std::make_shared<std::promise<void>>();
  std::shared_future<void> Fut = Done->get_future().share();
  std::thread([Done, Fn]() mutable {
    try {
      Fn();
      Done->set_value();
    } catch (...) {
      Done->set_exception(std::current_exception());
    }
  }).detach();
  return Fut;
}

static void finish(const std::shared_future<void> &Fut, const char *What) {
  if (Fut.wait_for(Timeout) != std::future_status::ready)
    fatal(What);
  Fut.get();
}

static void waitBounded(sycl::event Ev, const char *What) {
  finish(startAsync([Ev]() mutable { Ev.wait(); }), What);
}

static void drain(std::vector<sycl::queue> Queues) {
  finish(startAsync([Queues]() mutable {
           for (sycl::queue &Q : Queues)
             Q.wait();
         }),
         "draining the queues");
}

static bool isComplete(const sycl::event &Ev) {
  return Ev.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete;
}

using HostAcc = sycl::host_accessor<int, 1, sycl::access_mode::read_write>;

enum class Creation { Constructor, GetHostAccess };

static void run(Creation How, bool Resignal) {
  sycl::queue Q1{sycl::property::queue::in_order{}};
  sycl::context Ctx = Q1.get_context();
  sycl::device Dev = Q1.get_device();
  sycl::queue Q2{Ctx, Dev, sycl::property::queue::in_order{}};
  sycl::queue Q3{Ctx, Dev, sycl::property::queue::in_order{}};

  int *Mid = sycl::malloc_device<int>(N, Dev, Ctx);
  int *Res = sycl::malloc_host<int>(N, Ctx);
  int *Unrelated = sycl::malloc_device<int>(N, Dev, Ctx);

  {
    sycl::buffer<int, 1> Buf{sycl::range<1>{N}};
    // Declared after the buffer: released first on every exit path.
    std::optional<HostAcc> First;
    std::optional<HostAcc> Copy;
    if (How == Creation::Constructor)
      First.emplace(Buf);
    else
      First.emplace(Buf.get_host_access());
    for (size_t I = 0; I < N; ++I)
      (*First)[I] = static_cast<int>(I);
    Copy.emplace(*First);

    // Blocked by the host accessor.
    sycl::event K = Q1.submit([&](sycl::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access_mode::read_write>(CGH);
      CGH.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> I) {
        Mid[I] = Acc[I] * 2;
        Acc[I] += 1;
      });
    });
    sycl::event E = syclex::make_event(Ctx);
    syclex::enqueue_signal_event(Q1, E);
    syclex::enqueue_wait_event(Q2, E);
    sycl::event C = Q2.parallel_for(
        sycl::range<1>{N}, [=](sycl::id<1> I) { Res[I] = Mid[I] + 5; });

    if (Resignal) {
      Q3.parallel_for(sycl::range<1>{N},
                      [=](sycl::id<1> I) { Unrelated[I] = 9; });
      syclex::enqueue_signal_event(Q3, E);
      waitBounded(E, "new signal of E");
    }

    // One copy of the host accessor is still alive.
    First.reset();
    std::this_thread::sleep_for(Settle);
    check(!isComplete(K), "kernel ran while a host accessor copy is alive");
    check(!isComplete(C), "consumer of E ran while a host accessor copy is "
                          "alive (best effort)");
    if (!Resignal)
      check(!isComplete(E), "E completed while a host accessor copy is alive");

    Copy.reset();
    waitBounded(K, "kernel after the host accessor was released");
    waitBounded(C, "consumer of E after the host accessor was released");
    if (!Resignal)
      waitBounded(E, "E after the host accessor was released");
    checkAll(
        Res, [](size_t I) { return static_cast<int>(I) * 2 + 5; },
        "consumer data");

    drain({Q1, Q2, Q3});
    HostAcc Final{Buf};
    for (size_t I = 0; I < N; ++I) {
      if (Final[I] != static_cast<int>(I) + 1) {
        std::cerr << "FAILED: buffer data at " << I << " got " << Final[I]
                  << ", expected " << I + 1 << std::endl;
        ++Failures;
        break;
      }
    }
  }

  sycl::free(Mid, Ctx);
  sycl::free(Res, Ctx);
  sycl::free(Unrelated, Ctx);
}

int main() {
  try {
    for (Creation How : {Creation::Constructor, Creation::GetHostAccess}) {
      run(How, /*Resignal*/ false);
      run(How, /*Resignal*/ true);
    }
  } catch (sycl::exception &Ex) {
    std::cerr << "Unexpected exception: " << Ex.what() << std::endl;
    return 1;
  }
  return Failures ? 1 : 0;
}
