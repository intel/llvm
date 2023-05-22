// REQUIRES: level_zero, gpu, TEMPORARY_DISABLED
// Disabled as thread safety not yet implemented

// RUN: %clangxx -pthread -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %{run} %t.out

// Test updating a graph in a threaded situation.

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

  auto DataA2 = DataA;
  auto DataB2 = DataB;
  auto DataC2 = DataC;

  exp_ext::command_graph GraphA{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);
  T *PtrC = malloc_device<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.copy(DataC.data(), PtrC, Size);
  Queue.wait_and_throw();

  GraphA.begin_recording(Queue);
  run_kernels_usm(Queue, Size, PtrA, PtrB, PtrC);
  GraphA.end_recording();

  auto GraphExec = GraphA.finalize();

  exp_ext::command_graph GraphB{Queue.get_context(), Queue.get_device()};

  T *PtrA2 = malloc_device<T>(Size, Queue);
  T *PtrB2 = malloc_device<T>(Size, Queue);
  T *PtrC2 = malloc_device<T>(Size, Queue);

  Queue.copy(DataA2.data(), PtrA2, Size);
  Queue.copy(DataB2.data(), PtrB2, Size);
  Queue.copy(DataC2.data(), PtrC2, Size);
  Queue.wait_and_throw();

  GraphB.begin_recording(Queue);
  run_kernels_usm(Queue, Size, PtrA2, PtrB2, PtrC2);
  GraphB.end_recording();

  auto UpdateGraph = [&]() { GraphExec.update(GraphB); };

  std::vector<std::thread> Threads;
  Threads.reserve(NumThreads);

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads.emplace_back(UpdateGraph);
  }

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads[i].join();
  }

  free(PtrA, Queue);
  free(PtrB, Queue);
  free(PtrC, Queue);

  return 0;
}
