// REQUIRES: level_zero_v2_adapter
// RUN: %{build} -o %t.out
// RUN: env SYCL_QUEUE_THREAD_POOL_SIZE=8 %{run} %t.out

// XFAIL: *
// XFAIL-TRACKER: review-10-02 #5 (host task events cannot be re-signaled)

// tests-10-02 E11: 2D copy/fill host fallback with pitches and captured
// signals - re-signal of the event returned by the fallback.
//
// A host-to-host ext_oneapi_memcpy2d and a host-target ext_oneapi_fill2d are
// executed by a host task, so the event they return is a host event. The
// specification allows passing any event returned by a queue member function
// to enqueue_signal_event, but CheckEventForSignal goes through
// CheckEventForWait, which rejects every host event, so the re-signal throws
// errc::invalid. Each case holds the operation behind a host task gate,
// captures its returned event in a consumer, re-signals that event on another
// queue, and checks that the consumer keeps waiting for the 2D operation and
// that the rows, their padding and the guards are correct. The negative "still
// pending" checks are best effort.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/ext/oneapi/memcpy2d.hpp>
#include <sycl/properties/all_properties.hpp>

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

constexpr size_t Guard = 16;
constexpr int Init = -1;

// Height rows of Pitch ints in plain host memory with Guard ints before and
// after them; true if the first Width ints of every row hold Expected(Row,
// Col) and all other ints still hold Init.
static bool holds(const std::vector<int> &Storage, size_t Pitch, size_t Width,
                  size_t Height,
                  const std::function<int(size_t, size_t)> &Expected) {
  for (size_t I = 0; I < Storage.size(); ++I) {
    bool InRows = I >= Guard && I < Guard + Pitch * Height;
    size_t Row = InRows ? (I - Guard) / Pitch : 0;
    size_t Col = InRows ? (I - Guard) % Pitch : 0;
    int Want = (InRows && Col < Width) ? Expected(Row, Col) : Init;
    if (Storage[I] != Want)
      return false;
  }
  return true;
}

// Holds Op (a 2D host-fallback operation on Q1 which returns its event)
// behind a gate, captures the returned event in a consumer, re-signals it on
// Q2 and checks that the consumer still waits for Op.
static void runCase(sycl::queue &Q1, sycl::queue &Q2, sycl::queue &Qc,
                    const std::string &Case,
                    const std::function<sycl::event()> &Op,
                    const std::function<bool()> &Verify) {
  auto G = std::make_shared<Gate>();
  GateOpener Opener{G};
  Q1.submit([&](sycl::handler &CGH) { CGH.host_task([G] { G->wait(); }); });
  sycl::event E = Op();

  // 0: not run yet, 1: saw the result of Op, 2: saw something else.
  auto Seen = std::make_shared<std::atomic<int>>(0);
  sycl::event Consumer = Qc.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.host_task([=] { *Seen = Verify() ? 1 : 2; });
  });

  try {
    syclex::enqueue_signal_event(Q2, E);
    E.wait();
    check(isComplete(E), Case, "the new signal is not complete");
  } catch (const sycl::exception &Ex) {
    check(false, Case, "re-signaling the returned event threw");
    std::cerr << "  " << Ex.what() << std::endl;
  }

  // Best effort: give a wrongly released consumer time to run.
  std::this_thread::sleep_for(100ms);
  check(!isComplete(Consumer), Case, "the consumer did not stay pending");
  check(*Seen == 0, Case, "the consumer ran before the original work");

  G->open();
  waitBounded(Consumer, Case);
  Q1.wait();
  check(!G->TimedOut, Case, "the gate timed out");
  check(*Seen == 1, Case, "the consumer did not see the 2D result");
  check(Verify(), Case, "wrong data after the original work");
}

static int valueAt(size_t Row, size_t Col) {
  return static_cast<int>(Row * 1000 + Col + 1);
}

int main() {
  sycl::queue Q1{sycl::property::queue::in_order{}};
  sycl::context Ctx = Q1.get_context();
  sycl::device Dev = Q1.get_device();
  sycl::queue Q2{Ctx, Dev, sycl::property::queue::in_order{}};
  sycl::queue Qc{Ctx, Dev, sycl::property::queue::in_order{}};

  constexpr size_t Height = 4, Width = 13, SrcPitch = 20, DstPitch = 17;

  {
    std::vector<int> Src(2 * Guard + SrcPitch * Height, 0);
    for (size_t R = 0; R < Height; ++R)
      for (size_t C = 0; C < SrcPitch; ++C)
        Src[Guard + R * SrcPitch + C] = valueAt(R, C);
    std::vector<int> Dst(2 * Guard + DstPitch * Height, Init);
    runCase(
        Q1, Q2, Qc, "returned event of a host-to-host copy2d",
        [&] {
          return Q1.ext_oneapi_copy2d(Src.data() + Guard, SrcPitch,
                                      Dst.data() + Guard, DstPitch, Width,
                                      Height);
        },
        [&] { return holds(Dst, DstPitch, Width, Height, valueAt); });
  }

  {
    const int Pattern = 0x12345678;
    std::vector<int> Dst(2 * Guard + DstPitch * Height, Init);
    runCase(
        Q1, Q2, Qc, "returned event of a host-target fill2d",
        [&] {
          return Q1.ext_oneapi_fill2d(Dst.data() + Guard, DstPitch, Pattern,
                                      Width, Height);
        },
        [&] {
          return holds(Dst, DstPitch, Width, Height,
                       [&](size_t, size_t) { return Pattern; });
        });
  }

  Q2.wait();
  Qc.wait();
  return Failures ? 1 : 0;
}
