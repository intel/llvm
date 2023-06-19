// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %GPU_CHECK_PLACEHOLDER
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out %GPU_CHECK_PLACEHOLDER 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Expected fail as sycl streams aren't implemented yet
// XFAIL: *

// This test checks that we can use a stream when explicitly adding a
// command_graph node.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  using T = int;

  size_t WorkItems = 16;
  std::vector<T> DataIn(WorkItems);

  std::iota(DataIn.begin(), DataIn.end(), 1);

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *PtrIn = malloc_device<T>(WorkItems, Queue);
  Queue.copy(DataIn.data(), PtrIn, WorkItems);

  Graph.add([&](handler &CGH) {
    sycl::stream Out(WorkItems * 16, 16, CGH);
    CGH.parallel_for(range<1>(WorkItems), [=](item<1> id) {
      Out << "Val: " << PtrIn[id.get_linear_id()] << sycl::endl;
    });
  });

  auto GraphExec = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });

  Queue.wait_and_throw();

  Queue.copy(PtrIn, DataIn.data(), Size);

  free(PtrIn, Queue);

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
