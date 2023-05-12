// REQUIRES: level_zero, gpu, TEMPORARY_DISABLED
// Disabled as thread safety not yet implemented

// RUN: %clangxx -pthread -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Test finalizing and submitting a graph in a threaded situation.

#include "../graph_common.hpp"

#include <thread>

int main() {
  queue TestQueue;

  const unsigned NumThreads = std::thread::hardware_concurrency();

  auto RecordGraph = [&]() {
    exp_ext::command_graph Graph{TestQueue.get_context(),
                                 TestQueue.get_device()};
    try {
      Graph.begin_recording(TestQueue);
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
