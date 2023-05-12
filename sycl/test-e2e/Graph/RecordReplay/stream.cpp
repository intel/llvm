// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER

// Expected fail as sycl::stream is not implemented yet
// XFAIL: *

// This test checks that we can use a stream within a command_graph recording.

#include "../graph_common.hpp"

int main() {
  queue TestQueue;

  using T = int;

  size_t WorkItems = 16;
  std::vector<T> DataIn(WorkItems);

  std::iota(DataIn.begin(), DataIn.end(), 1);

  exp_ext::command_graph Graph{TestQueue.get_context(), TestQueue.get_device()};

  T *PtrIn = malloc_device<T>(WorkItems, TestQueue);
  TestQueue.copy(DataIn.data(), PtrIn, WorkItems);

  Graph.begin_recording(TestQueue);

  TestQueue.submit([&](handler &CGH) {
    sycl::stream Out(WorkItems * 16, 16, CGH);
    CGH.parallel_for(range<1>(WorkItems), [=](item<1> id) {
      Out << "Val: " << PtrIn[id.get_linear_id()] << sycl::endl;
    });
  });
  Graph.end_recording();

  auto GraphExec = Graph.finalize();

  TestQueue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });

  TestQueue.wait_and_throw();

  TestQueue.copy(PtrIn, DataIn.data(), Size);

  free(PtrIn, TestQueue);

  return 0;
}

// CHECK-DAG: Val: 1
// CHECK-DAG: Val: 2
// CHECK-DAG: Val: 3
// CHECK-DAG: Val: 4
// CHECK-DAG: Val: 5
// CHECK-DAG: Val: 6
// CHECK-DAG: Val: 7
// CHECK-DAG: Val: 8
// CHECK-DAG: Val: 9
// CHECK-DAG: Val: 10
// CHECK-DAG: Val: 11
// CHECK-DAG: Val: 12
// CHECK-DAG: Val: 13
// CHECK-DAG: Val: 14
// CHECK-DAG: Val: 15
// CHECK-DAG: Val: 16
