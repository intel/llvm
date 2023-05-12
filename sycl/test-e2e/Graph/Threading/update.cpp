// REQUIRES: level_zero, gpu, TEMPORARY_DISABLED
// Disabled as thread safety not yet implemented

// RUN: %clangxx -pthread -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Test updating a graph in a threaded situation.

#include "../graph_common.hpp"

#include <thread>

int main() {
  queue TestQueue;

  using T = int;

  const unsigned NumThreads = std::thread::hardware_concurrency();
  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  auto DataA2 = DataA;
  auto DataB2 = DataB;
  auto DataC2 = DataC;

  exp_ext::command_graph GraphA{TestQueue.get_context(),
                                TestQueue.get_device()};

  T *PtrA = malloc_device<T>(Size, TestQueue);
  T *PtrB = malloc_device<T>(Size, TestQueue);
  T *PtrC = malloc_device<T>(Size, TestQueue);

  TestQueue.copy(DataA.data(), PtrA, Size);
  TestQueue.copy(DataB.data(), PtrB, Size);
  TestQueue.copy(DataC.data(), PtrC, Size);
  TestQueue.wait_and_throw();

  GraphA.begin_recording(TestQueue);
  run_kernels_usm(TestQueue, Size, PtrA, PtrB, PtrC);
  GraphA.end_recording();

  auto GraphExec = GraphA.finalize();

  exp_ext::command_graph GraphB{TestQueue.get_context(),
                                TestQueue.get_device()};

  T *PtrA2 = malloc_device<T>(Size, TestQueue);
  T *PtrB2 = malloc_device<T>(Size, TestQueue);
  T *PtrC2 = malloc_device<T>(Size, TestQueue);

  TestQueue.copy(DataA2.data(), PtrA2, Size);
  TestQueue.copy(DataB2.data(), PtrB2, Size);
  TestQueue.copy(DataC2.data(), PtrC2, Size);
  TestQueue.wait_and_throw();

  GraphB.begin_recording(TestQueue);
  run_kernels_usm(TestQueue, Size, PtrA2, PtrB2, PtrC2);
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

  free(PtrA, TestQueue);
  free(PtrB, TestQueue);
  free(PtrC, TestQueue);

  return 0;
}
