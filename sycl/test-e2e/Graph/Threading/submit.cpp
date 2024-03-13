// RUN: %{build_pthread_inc} -o %t.out
// RUN: %{run} %t.out
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero && linux %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test submitting a graph in a threaded situation.
// Note that we do not check the outputs because multiple concurrent executions
// is indeterministic (and depends on the backend command management).
// However, this test verifies that concurrent graph submissions do not trigger
// errors nor memory leaks.

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

  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  calculate_reference_data(NumThreads, Size, ReferenceA, ReferenceB,
                           ReferenceC);

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);
  T *PtrC = malloc_device<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.copy(DataC.data(), PtrC, Size);
  Queue.wait_and_throw();

  Graph.begin_recording(Queue);
  run_kernels_usm(Queue, Size, PtrA, PtrB, PtrC);
  Graph.end_recording();

  std::vector<exp_ext::command_graph<exp_ext::graph_state::executable>>
      GraphExecs;
  for (unsigned i = 0; i < NumThreads; ++i) {
    GraphExecs.push_back(Graph.finalize());
  }

  Barrier SyncPoint{NumThreads};

  auto SubmitGraph = [&](int ThreadNum) {
    SyncPoint.wait();
    Queue.submit([&](sycl::handler &CGH) {
      CGH.ext_oneapi_graph(GraphExecs[ThreadNum]);
    });
  };

  std::vector<std::thread> Threads;
  Threads.reserve(NumThreads);

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads.emplace_back(SubmitGraph, i);
  }

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads[i].join();
  }

  Queue.wait_and_throw();

  free(PtrA, Queue);
  free(PtrB, Queue);
  free(PtrC, Queue);

  return 0;
}
