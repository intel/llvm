// REQUIRES: level_zero, gpu, TEMPORARY_DISABLED
// Disabled as thread safety not yet implemented

// RUN: %clangxx -pthread -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Test finalizing and submitting a graph in a threaded situation.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"

#include <thread>

int main() {
  queue Queue;

  const unsigned NumThreads = std::thread::hardware_concurrency();

  auto RecordGraph = [&]() {
    exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
    try {
      Graph.begin_recording(Queue);
    } catch (sycl::exception &E) {
      // Can throw if graph is already being recorded to
    }
    Graph.end_recording();
  };

  std::vector<std::thread> Threads;
  Threads.reserve(NumThreads);
  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads.emplace_back(RecordGraph);
  }

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads[i].join();
  }

  return 0;
}
