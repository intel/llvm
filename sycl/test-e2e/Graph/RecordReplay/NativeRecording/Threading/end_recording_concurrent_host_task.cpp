// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} %threads_lib -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests that ending native recording on a queue does not deadlock against a
// concurrent restricted host task submission to that same queue. Ending capture
// takes the queue lock, while submitting a captured host task takes the queue
// lock and then the graph lock, so end_recording(queue) must not hold the graph
// lock while it ends capture.

#include "../../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>

#include <atomic>
#include <chrono>
#include <thread>

constexpr int RaceAttempts = 50;

int main() {
  queue Queue{property::queue::in_order{}};

  constexpr size_t N = 16;
  uint32_t *Data = malloc_shared<uint32_t>(N, Queue);
  std::fill(Data, Data + N, 0);

  for (int I = 0; I < RaceAttempts; ++I) {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::enable_native_recording{}}};

    Graph.begin_recording(Queue);

    std::atomic<bool> Stop{false};
    std::thread Submitter{[&] {
      while (!Stop.load()) {
        exp_ext::host_task(Queue, [=] { Data[0] += 1; });
      }
    }};

    // Give the submitter time to get into the submission path so that
    // end_recording lands in the middle of it.
    std::this_thread::sleep_for(std::chrono::microseconds(10));

    Graph.end_recording(Queue);

    Stop.store(true);
    Submitter.join();

    Queue.wait();
  }

  free(Data, Queue);

  return 0;
}
