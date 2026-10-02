// REQUIRES: level_zero_v2_adapter
// REQUIRES: aspect-usm_device_allocations
// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// tests-10-02 E11: 2D copy/fill host fallback with pitches and captured
// signals.
//
// Host-to-host 2D copies and 2D fills/memsets of host memory are executed by a
// host task, also on backends with native 2D support. Each case holds such an
// operation behind a host task gate on an in-order queue and signals an event
// E after it. A consumer on another queue captures E and checks the
// destination, then E is re-signaled on a third queue and that signal
// completes. The consumer must keep waiting for the original signal, and the
// destination must hold the rectangle while its row padding and the guard
// regions around it stay untouched. A device-target 2D copy is the native
// control: its returned event is re-signaled directly. The negative "still
// pending" checks are best effort.
//
// Re-signaling the event returned by the host fallback itself is covered by
// memcpy2d_host_fallback_resignal_returned_event.cpp.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/ext/oneapi/memcpy2d.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;
using namespace std::chrono_literals;

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

// Polls E until it completes. A missing completion exits with a failure
// instead of hanging, since a pending host task may still use local data.
static void waitBounded(const sycl::event &E, const std::string &Case) {
  auto Deadline = std::chrono::steady_clock::now() + 60s;
  while (!isComplete(E)) {
    if (std::chrono::steady_clock::now() > Deadline) {
      std::cerr << "FAILED (" << Case << "): no completion before the deadline"
                << std::endl;
      std::_Exit(1);
    }
    std::this_thread::sleep_for(1ms);
  }
}

// Rows of Height x Pitch elements in plain host memory (not USM), with guard
// elements before and after them. Width is the width of the rectangle.
template <typename T> struct Pitched2D {
  static constexpr size_t Guard = 16;
  size_t Pitch, Width, Height;
  std::vector<T> Storage;

  Pitched2D(size_t Pitch, size_t Width, size_t Height, T Init)
      : Pitch(Pitch), Width(Width), Height(Height),
        Storage(2 * Guard + Pitch * Height, Init) {}

  T *rows() { return Storage.data() + Guard; }
  T &at(size_t Row, size_t Col) { return rows()[Row * Pitch + Col]; }

  // True if the rectangle holds Expected(Row, Col) and every other element,
  // including the row padding and the guards, still holds Init.
  bool holds(const std::function<T(size_t, size_t)> &Expected, T Init) const {
    for (size_t I = 0; I < Storage.size(); ++I) {
      bool InRows = I >= Guard && I < Guard + Pitch * Height;
      size_t Row = InRows ? (I - Guard) / Pitch : 0;
      size_t Col = InRows ? (I - Guard) % Pitch : 0;
      T Want = (InRows && Col < Width) ? Expected(Row, Col) : Init;
      if (Storage[I] != Want)
        return false;
    }
    return true;
  }
};

struct Queues {
  sycl::queue Q1; // the gated original work
  sycl::queue Q2; // the new signal
  sycl::queue Qc; // the consumer
};

// Holds Op (a 2D operation on Q1) behind a closed gate, signals E after it,
// captures E in a consumer, re-signals E on Q2 and checks that the consumer
// sees the result of Op only after the gate opens.
static void runCase(Queues &Qs, const std::string &Case,
                    const std::function<void()> &Op,
                    const std::function<bool()> &Verify) {
  auto G = std::make_shared<Gate>();
  GateOpener Opener{G};
  sycl::event E = syclex::make_event(Qs.Q1.get_context());

  Qs.Q1.submit([&](sycl::handler &CGH) { CGH.host_task([G] { G->wait(); }); });
  Op();
  syclex::enqueue_signal_event(Qs.Q1, E);

  // 0: not run yet, 1: saw the result of Op, 2: saw something else.
  auto Seen = std::make_shared<std::atomic<int>>(0);
  sycl::event Consumer = Qs.Qc.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.host_task([=] { *Seen = Verify() ? 1 : 2; });
  });

  syclex::enqueue_signal_event(Qs.Q2, E);
  E.wait();
  check(isComplete(E), Case, "the new signal is not complete");

  // Best effort: give a wrongly released consumer time to run.
  std::this_thread::sleep_for(100ms);
  check(!isComplete(Consumer), Case, "the consumer did not stay pending");
  check(*Seen == 0, Case, "the consumer ran before the original work");

  G->open();
  waitBounded(Consumer, Case);
  Qs.Q1.wait();
  check(!G->TimedOut, Case, "the gate timed out");
  check(*Seen == 1, Case, "the consumer did not see the 2D result");
  check(Verify(), Case, "wrong data after the original work");
}

static unsigned char byteAt(size_t Row, size_t Col) {
  return static_cast<unsigned char>(Row * 37 + Col + 1);
}
static int intAt(size_t Row, size_t Col) {
  return static_cast<int>(Row * 1000 + Col + 1);
}

// Byte copies with ext_oneapi_memcpy2d.
static void memcpyCase(Queues &Qs, bool Handler, size_t Height) {
  std::string Case = std::string(Handler ? "handler" : "queue") +
                     " memcpy2d, height " + std::to_string(Height);
  constexpr unsigned char Init = 0xEE;
  Pitched2D<unsigned char> Src(40, 24, Height, 0);
  for (size_t R = 0; R < Height; ++R)
    for (size_t C = 0; C < Src.Pitch; ++C)
      Src.at(R, C) = byteAt(R, C);
  Pitched2D<unsigned char> Dst(48, 24, Height, Init);

  auto Op = [&] {
    if (Handler)
      Qs.Q1.submit([&](sycl::handler &CGH) {
        CGH.ext_oneapi_memcpy2d(Dst.rows(), Dst.Pitch, Src.rows(), Src.Pitch,
                                Dst.Width, Height);
      });
    else
      Qs.Q1.ext_oneapi_memcpy2d(Dst.rows(), Dst.Pitch, Src.rows(), Src.Pitch,
                                Dst.Width, Height);
  };
  runCase(Qs, Case, Op, [&] { return Dst.holds(byteAt, Init); });
}

// Typed copies with ext_oneapi_copy2d.
static void copyCase(Queues &Qs, bool Handler, size_t Height) {
  std::string Case = std::string(Handler ? "handler" : "queue") +
                     " copy2d, height " + std::to_string(Height);
  constexpr int Init = -1;
  Pitched2D<int> Src(20, 13, Height, 0);
  for (size_t R = 0; R < Height; ++R)
    for (size_t C = 0; C < Src.Pitch; ++C)
      Src.at(R, C) = intAt(R, C);
  Pitched2D<int> Dst(17, 13, Height, Init);

  auto Op = [&] {
    if (Handler)
      Qs.Q1.submit([&](sycl::handler &CGH) {
        CGH.ext_oneapi_copy2d(Src.rows(), Src.Pitch, Dst.rows(), Dst.Pitch,
                              Dst.Width, Height);
      });
    else
      Qs.Q1.ext_oneapi_copy2d(Src.rows(), Src.Pitch, Dst.rows(), Dst.Pitch,
                              Dst.Width, Height);
  };
  runCase(Qs, Case, Op, [&] { return Dst.holds(intAt, Init); });
}

// Typed fills of host memory with ext_oneapi_fill2d.
static void fillCase(Queues &Qs, bool Handler, size_t Height) {
  std::string Case = std::string(Handler ? "handler" : "queue") +
                     " fill2d, height " + std::to_string(Height);
  constexpr int Init = -1;
  const int Pattern = 0x12345678;
  Pitched2D<int> Dst(19, 11, Height, Init);

  auto Op = [&] {
    if (Handler)
      Qs.Q1.submit([&](sycl::handler &CGH) {
        CGH.ext_oneapi_fill2d(Dst.rows(), Dst.Pitch, Pattern, Dst.Width,
                              Height);
      });
    else
      Qs.Q1.ext_oneapi_fill2d(Dst.rows(), Dst.Pitch, Pattern, Dst.Width,
                              Height);
  };
  runCase(Qs, Case, Op, [&] {
    return Dst.holds([&](size_t, size_t) { return Pattern; }, Init);
  });
}

// Byte fills of host memory with ext_oneapi_memset2d.
static void memsetCase(Queues &Qs, bool Handler, size_t Height) {
  std::string Case = std::string(Handler ? "handler" : "queue") +
                     " memset2d, height " + std::to_string(Height);
  constexpr unsigned char Init = 0xEE;
  constexpr unsigned char Value = 0x5A;
  Pitched2D<unsigned char> Dst(56, 33, Height, Init);

  auto Op = [&] {
    if (Handler)
      Qs.Q1.submit([&](sycl::handler &CGH) {
        CGH.ext_oneapi_memset2d(Dst.rows(), Dst.Pitch, Value, Dst.Width,
                                Height);
      });
    else
      Qs.Q1.ext_oneapi_memset2d(Dst.rows(), Dst.Pitch, Value, Dst.Width,
                                Height);
  };
  runCase(Qs, Case, Op, [&] {
    return Dst.holds([](size_t, size_t) { return Value; }, Init);
  });
}

// A zero-height copy is documented to do nothing: the destination, including
// the rows it would have touched, must stay untouched.
static void zeroSizeCase(Queues &Qs) {
  constexpr unsigned char Init = 0xEE;
  Pitched2D<unsigned char> Src(40, 24, 2, 1);
  Pitched2D<unsigned char> Dst(48, 0, 2, Init);
  auto Op = [&] {
    Qs.Q1.ext_oneapi_memcpy2d(Dst.rows(), Dst.Pitch, Src.rows(), Src.Pitch, 24,
                              0);
  };
  runCase(Qs, "queue memcpy2d, height 0", Op,
          [&] { return Dst.holds(byteAt, Init); });
}

// Native control: a host-to-device 2D copy does not use the host fallback, so
// its returned event can be re-signaled directly.
static void deviceTargetControl(Queues &Qs) {
  const std::string Case = "device-target memcpy2d control";
  constexpr size_t Pitch = 64, Width = 48, Height = 4;
  std::vector<unsigned char> Src(Pitch * Height);
  for (size_t I = 0; I < Src.size(); ++I)
    Src[I] = byteAt(I / Pitch, I % Pitch);
  unsigned char *Dev =
      sycl::malloc_device<unsigned char>(Pitch * Height, Qs.Q1);
  Qs.Q1.memset(Dev, 0xEE, Pitch * Height).wait();
  std::vector<unsigned char> Out(Pitch * Height, 0);

  auto G = std::make_shared<Gate>();
  GateOpener Opener{G};
  Qs.Q1.submit([&](sycl::handler &CGH) { CGH.host_task([G] { G->wait(); }); });
  sycl::event E =
      Qs.Q1.ext_oneapi_memcpy2d(Dev, Pitch, Src.data(), Pitch, Width, Height);
  sycl::event Consumer = Qs.Qc.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.memcpy(Out.data(), Dev, Pitch * Height);
  });

  syclex::enqueue_signal_event(Qs.Q2, E);
  E.wait();
  // Best effort: give a wrongly released consumer time to run.
  std::this_thread::sleep_for(100ms);
  check(!isComplete(Consumer), Case, "the consumer did not stay pending");

  G->open();
  waitBounded(Consumer, Case);
  Qs.Q1.wait();
  bool Ok = true;
  for (size_t I = 0; I < Out.size(); ++I) {
    size_t Col = I % Pitch;
    unsigned char Want = Col < Width ? byteAt(I / Pitch, Col) : 0xEE;
    Ok = Ok && Out[I] == Want;
  }
  check(Ok, Case, "wrong data after the original work");
  sycl::free(Dev, Qs.Q1);
}

int main() {
  sycl::queue Q1{sycl::property::queue::in_order{}};
  sycl::context Ctx = Q1.get_context();
  sycl::device Dev = Q1.get_device();
  Queues Qs{Q1, sycl::queue{Ctx, Dev, sycl::property::queue::in_order{}},
            sycl::queue{Ctx, Dev, sycl::property::queue::in_order{}}};

  for (bool Handler : {false, true}) {
    for (size_t Height : {1, 5}) {
      memcpyCase(Qs, Handler, Height);
      copyCase(Qs, Handler, Height);
      fillCase(Qs, Handler, Height);
      memsetCase(Qs, Handler, Height);
    }
  }
  zeroSizeCase(Qs);
  deviceTargetControl(Qs);

  Qs.Q2.wait();
  Qs.Qc.wait();
  return Failures ? 1 : 0;
}
