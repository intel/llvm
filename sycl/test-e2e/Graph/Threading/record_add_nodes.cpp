// REQUIRES: level_zero, gpu, TEMPORARY_DISABLED
// Disabled as thread safety not yet implemented

// RUN: %clangxx -pthread -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %{run} %t.out

// Test recording commands to a queue in a threaded situation. We don't
// submit the graph to verify the results as ordering of graph nodes isn't
// defined.

#include "../graph_common.hpp"
#include <thread>

int main() {
  queue Queue;

  using T = int;

  const unsigned NumThreads = std::thread::hardware_concurrency();
  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);
  T *PtrC = malloc_device<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.copy(DataC.data(), PtrC, Size);
  Queue.wait_and_throw();

  Graph.begin_recording(Queue);
  auto recordGraph = [&]() {
    // Record commands to graph
    run_kernels_usm(Queue, Size, PtrA, PtrB, PtrC);
  };
  Graph.end_recording();

  std::vector<std::thread> Threads;
  Threads.reserve(NumThreads);
  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads.emplace_back(recordGraph);
  }

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads[i].join();
  }

  free(PtrA, Queue);
  free(PtrB, Queue);
  free(PtrC, Queue);

  return 0;
}
