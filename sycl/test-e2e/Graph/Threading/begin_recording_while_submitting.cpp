// RUN: %{build} %threads_lib -o %t.out
// RUN: %{run} %t.out

// Regression test for an AB-BA deadlock between the graph mutex and the queue's
// submission mutex


#include "../graph_common.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <thread>

int main() {
  queue Q1;
  queue Q2{Q1.get_context(), Q1.get_device()};

  exp_ext::command_graph Graph{Q1.get_context(), Q1.get_device()};

  Graph.begin_recording(Q1);
  sycl::event GraphEvent =
      Q1.submit([&](handler &CGH) { CGH.single_task([=]() {}); });
  Graph.end_recording(Q1);

  constexpr int MaxIterations = 20000;

  // how many consecutive no-progress one-second samples before we declare a deadlock
  constexpr int StallLimit = 10;

  // so we don't spin forever
  constexpr auto Budget = std::chrono::seconds(10);
  const auto Deadline = std::chrono::steady_clock::now() + Budget;

  std::atomic<int> ProgressA{0}, ProgressB{0};
  std::atomic<bool> Done{false};

  std::thread A([&] {
    for (int I = 0; I < MaxIterations; ++I) {
      if (std::chrono::steady_clock::now() > Deadline)
        break;
      try {
        Graph.begin_recording(Q2);
        Graph.end_recording(Q2);
      } catch (sycl::exception &) {
        // just ignore these for this test
      }
      ProgressA.store(I, std::memory_order_relaxed);
    }
  });

  std::thread B([&] {
    for (int I = 0; I < MaxIterations; ++I) {
      if (std::chrono::steady_clock::now() > Deadline)
        break;
      try {
        Q2.ext_oneapi_submit_barrier({GraphEvent});
      } catch (sycl::exception &) {
        // just ignore these for this test
      }
      ProgressB.store(I, std::memory_order_relaxed);
    }
  });

  std::thread Watchdog([&] {
    int LastA = -1, LastB = -1, Stalls = 0;
    while (!Done.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      const int CurA = ProgressA.load(std::memory_order_relaxed);
      const int CurB = ProgressB.load(std::memory_order_relaxed);
      if (CurA == LastA && CurB == LastB) {
        if (++Stalls == StallLimit) {
          std::cerr << "Deadlock: no progress in " << StallLimit
                    << "s (A=" << CurA << " B=" << CurB << ")" << std::endl;
          std::abort();
        }
      } else {
        Stalls = 0;
      }
      LastA = CurA;
      LastB = CurB;
    }
  });

  A.join();
  B.join();
  Done.store(true, std::memory_order_relaxed);
  Watchdog.join();

  return 0;
}
