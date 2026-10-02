// REQUIRES: level_zero_v2_adapter
// REQUIRES: aspect-usm_device_allocations, aspect-usm_host_allocations

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// tests-10-02 E01 (control): the default constructed event scenario of
// default_constructed_signal.cpp with an event from an explicit-context
// make_event. Signal it, consume the signal on a second queue of that context,
// re-signal it through an alias, and destroy the event, the queues and a
// consumer event in different orders.

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

namespace syclex = sycl::ext::oneapi::experimental;
using namespace std::chrono_literals;

constexpr size_t N = 1024;
// Bound for anything that must finish: a missing wakeup fails, not hangs.
constexpr auto Timeout = 30s;

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

static bool isComplete(const sycl::event &Ev) {
  return Ev.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete;
}

static void run(bool DestroyEventFirst) {
  sycl::device Dev;
  sycl::context Ctx = Dev.get_platform().khr_get_default_context();

  std::optional<sycl::event> E;
  E.emplace(syclex::make_event(Ctx));
  check(isComplete(*E), "a new event from make_event is complete");

  std::optional<sycl::queue> Q1, Q2;
  Q1.emplace(Ctx, Dev, sycl::property::queue::in_order{});
  Q2.emplace(Ctx, Dev, sycl::property::queue::in_order{});

  int *Data = sycl::malloc_device<int>(N, Dev, Ctx);
  int *Res = sycl::malloc_host<int>(N, Ctx);
  sycl::event Consumer;

  // First signal. Q2 is ordered after Q1's kernel only through E.
  Q1->parallel_for(sycl::range<1>{N}, [=](sycl::id<1> I) {
    Data[I] = static_cast<int>(I[0]) + 1;
  });
  syclex::enqueue_signal_event(*Q1, *E);
  syclex::enqueue_wait_event(*Q2, *E);
  Consumer = Q2->parallel_for(sycl::range<1>{N},
                              [=](sycl::id<1> I) { Res[I] = Data[I] * 2; });
  waitBounded(Consumer, "consumer of the first signal");
  waitBounded(*E, "first signal of the event");
  check(isComplete(*E), "first signal reports complete");
  checkAll(
      Res, [](size_t I) { return (static_cast<int>(I) + 1) * 2; },
      "data after the first signal");

  // Second signal through an alias, consumed through a vector wait.
  {
    sycl::event Alias = *E;
    Q1->parallel_for(sycl::range<1>{N}, [=](sycl::id<1> I) { Data[I] += 10; });
    syclex::enqueue_signal_event(*Q1, Alias);
    syclex::enqueue_wait_events(*Q2, {Alias});
    Consumer = Q2->parallel_for(sycl::range<1>{N},
                                [=](sycl::id<1> I) { Res[I] = Data[I] * 3; });
    waitBounded(Consumer, "consumer of the second signal");
    waitBounded(*E, "second signal observed through the original handle");
    check(isComplete(Alias), "second signal reports complete");
    checkAll(
        Res, [](size_t I) { return (static_cast<int>(I) + 11) * 3; },
        "data after the second signal");
  }

  sycl::free(Data, Ctx);
  sycl::free(Res, Ctx);

  // Leave the scopes in different orders; the consumer event goes last.
  if (DestroyEventFirst) {
    E.reset();
    Q1.reset();
    Q2.reset();
  } else {
    Q2.reset();
    Q1.reset();
    E.reset();
  }
}

int main() {
  try {
    run(/*DestroyEventFirst*/ true);
    run(/*DestroyEventFirst*/ false);
  } catch (sycl::exception &Ex) {
    std::cerr << "Unexpected exception: " << Ex.what() << std::endl;
    return 1;
  }
  return Failures ? 1 : 0;
}
