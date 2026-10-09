// REQUIRES: level_zero_v2_adapter
// REQUIRES: aspect-usm_device_allocations, aspect-usm_host_allocations

// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// tests-10-02 E06: Returned events of different command types can be reused
// before the original command completes.
//
// For each command type, on an in-order and on an out-of-order queue: a gated
// handler::host_task is followed by the command, which returns E. A consumer
// on another queue captures E, an eventless command is submitted on the same
// queue, and E is then re-signaled on a third queue behind unrelated work, and
// the new signal completes. The consumer and a wait on the original queue must
// still follow the original command, and the data must be correct. The prefetch
// and mem_advise commands use shared USM and are skipped without it. The
// zero-size memcpy is a no-op, so only completion is checked for it.
// Corresponds to unit scenario U12.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
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

// Holds a host task until the test opens it. The host task side is bounded.
struct Gate {
  std::mutex M;
  std::condition_variable CV;
  bool IsOpen = false;
  std::atomic<bool> Entered{false};
  std::atomic<bool> TimedOut{false};

  // Called by the host task.
  void pass() {
    Entered = true;
    std::unique_lock<std::mutex> Lock(M);
    if (!CV.wait_for(Lock, Timeout, [this] { return IsOpen; }))
      TimedOut = true;
  }
  void open() {
    {
      std::lock_guard<std::mutex> Lock(M);
      IsOpen = true;
    }
    CV.notify_all();
  }
  void waitEntered() {
    auto Deadline = std::chrono::steady_clock::now() + Timeout;
    while (!Entered) {
      if (std::chrono::steady_clock::now() > Deadline)
        fatal("a host task did not reach its gate");
      std::this_thread::sleep_for(1ms);
    }
  }
};
using GatePtr = std::shared_ptr<Gate>;

// Opens the gates on every exit path, including exceptions.
struct OpenOnExit {
  std::vector<GatePtr> Gates;
  ~OpenOnExit() {
    for (const GatePtr &G : Gates)
      G->open();
  }
};

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

static bool hasReturned(const std::shared_future<void> &Fut) {
  return Fut.wait_for(0s) == std::future_status::ready;
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

enum class Op { Kernel, Memcpy, Fill, Memset, Prefetch, MemAdvise, ZeroSize };

static const char *nameOf(Op O) {
  switch (O) {
  case Op::Kernel:
    return "kernel";
  case Op::Memcpy:
    return "memcpy";
  case Op::Fill:
    return "fill";
  case Op::Memset:
    return "memset";
  case Op::Prefetch:
    return "prefetch";
  case Op::MemAdvise:
    return "mem_advise";
  case Op::ZeroSize:
    return "zero-size memcpy";
  }
  return "?";
}

static bool usesShared(Op O) { return O == Op::Prefetch || O == Op::MemAdvise; }

// The value of Data[I] after the command.
static int expectedData(Op O, size_t I) {
  switch (O) {
  case Op::Kernel:
    return (static_cast<int>(I) + 1) * 2;
  case Op::Memcpy:
    return 100 + static_cast<int>(I);
  case Op::Fill:
    return 42;
  case Op::Memset:
    return 0x01010101;
  default:
    return 5; // Prefilled, not changed by prefetch or mem_advise.
  }
}

static void run(Op O, bool InOrder) {
  std::cerr << "Running " << nameOf(O)
            << (InOrder ? ", in-order" : ", out-of-order") << std::endl;
  sycl::device Dev;
  sycl::context Ctx{Dev};
  sycl::queue Q1 =
      InOrder ? sycl::queue{Ctx, Dev, sycl::property::queue::in_order{}}
              : sycl::queue{Ctx, Dev};
  sycl::queue Q2{Ctx, Dev, sycl::property::queue::in_order{}};
  sycl::queue Q3{Ctx, Dev, sycl::property::queue::in_order{}};

  int *Marker = sycl::malloc_host<int>(N, Ctx); // Written by the host task.
  int *Src = sycl::malloc_host<int>(N, Ctx);    // Written by the host task.
  int *Data = usesShared(O) ? sycl::malloc_shared<int>(N, Dev, Ctx)
                            : sycl::malloc_device<int>(N, Dev, Ctx);
  int *Res = sycl::malloc_host<int>(N, Ctx);
  int *After = sycl::malloc_host<int>(N, Ctx);
  int *Unrelated = sycl::malloc_device<int>(N, Dev, Ctx);
  if (usesShared(O))
    for (size_t I = 0; I < N; ++I)
      Data[I] = 5;
  for (size_t I = 0; I < N; ++I)
    Marker[I] = 0;

  auto G = std::make_shared<Gate>();
  OpenOnExit Opener{{G}};

  sycl::event H = Q1.submit([&](sycl::handler &CGH) {
    CGH.host_task([G, Marker, Src] {
      G->pass();
      for (size_t I = 0; I < N; ++I) {
        Marker[I] = 1;
        Src[I] = 100 + static_cast<int>(I);
      }
    });
  });
  G->waitEntered();

  sycl::event E = Q1.submit([&](sycl::handler &CGH) {
    if (!InOrder)
      CGH.depends_on(H);
    switch (O) {
    case Op::Kernel:
      CGH.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> I) {
        Data[I] = (Marker[I] + static_cast<int>(I)) * 2;
      });
      break;
    case Op::Memcpy:
      CGH.memcpy(Data, Src, N * sizeof(int));
      break;
    case Op::Fill:
      CGH.fill<int>(Data, 42, N);
      break;
    case Op::Memset:
      CGH.memset(Data, 1, N * sizeof(int));
      break;
    case Op::Prefetch:
      CGH.prefetch(Data, N * sizeof(int));
      break;
    case Op::MemAdvise:
      CGH.mem_advise(Data, N * sizeof(int), 0);
      break;
    case Op::ZeroSize:
      CGH.memcpy(Data, Src, 0);
      break;
    }
  });

  // A consumer which captures E.
  const bool CheckData = O != Op::ZeroSize;
  sycl::event Consumer = Q2.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> I) {
      Res[I] = CheckData ? Data[I] + Marker[I] * 1000 : 7;
    });
  });

  // An eventless command on the original queue.
  if (InOrder) {
    syclex::parallel_for(Q1, sycl::range<1>{N},
                         [=](sycl::id<1> I) { After[I] = Marker[I] + 1; });
  } else {
    syclex::submit(Q1, [&](sycl::handler &CGH) {
      CGH.depends_on(H);
      CGH.parallel_for(sycl::range<1>{N},
                       [=](sycl::id<1> I) { After[I] = Marker[I] + 1; });
    });
  }

  // Reuse E before the original command completes.
  Q3.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> I) { Unrelated[I] = 9; });
  syclex::enqueue_signal_event(Q3, E);
  waitBounded(E, "new signal of the returned event");

  std::shared_future<void> Q1Wait = startAsync([Q1]() mutable { Q1.wait(); });
  std::this_thread::sleep_for(Settle);
  if (CheckData)
    check(!isComplete(Consumer), "consumer of the original command followed "
                                 "the new signal (best effort)");
  check(!hasReturned(Q1Wait), "wait on the original queue returned with its "
                              "host task gated (best effort)");

  G->open();
  finish(Q1Wait, "wait on the original queue");
  waitBounded(Consumer, "consumer of the original command");
  if (CheckData)
    checkAll(
        Res, [O](size_t I) { return expectedData(O, I) + 1000; },
        "consumer data");
  else
    checkAll(
        Res, [](size_t) { return 7; }, "consumer of the no-op");
  checkAll(
      After, [](size_t) { return 2; }, "eventless command data");

  drain({Q1, Q2, Q3});
  check(!G->TimedOut, "a host task gate timed out");
  for (int *P : {Marker, Src, Data, Res, After, Unrelated})
    sycl::free(P, Ctx);
}

int main() {
  try {
    sycl::device Dev;
    const bool HasShared = Dev.has(sycl::aspect::usm_shared_allocations);
    for (Op O : {Op::Kernel, Op::Memcpy, Op::Fill, Op::Memset, Op::Prefetch,
                 Op::MemAdvise, Op::ZeroSize}) {
      if (usesShared(O) && !HasShared)
        continue;
      run(O, /*InOrder*/ true);
      run(O, /*InOrder*/ false);
    }
  } catch (sycl::exception &Ex) {
    std::cerr << "Unexpected exception: " << Ex.what() << std::endl;
    return 1;
  }
  return Failures ? 1 : 0;
}
