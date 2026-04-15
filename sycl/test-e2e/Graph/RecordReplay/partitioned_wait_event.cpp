// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests that event::wait() during recording with allow_wait_recording property
// creates partition boundaries, and that the resulting partitioned graph
// executes correctly.

#include "../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue{property::queue::in_order{}};

  exp_ext::command_graph Graph{
      Queue.get_context(), Queue.get_device(),
      {exp_ext::property::graph::allow_wait_recording{}}};

  const size_t N = 100;
  int *A = malloc_device<int>(N, Queue);
  int *B = malloc_device<int>(N, Queue);
  int *C = malloc_device<int>(N, Queue);

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      A[it] = static_cast<int>(it);
      B[it] = 0;
      C[it] = 0;
    });
  }).wait();

  Graph.begin_recording(Queue);

  // Part 1
  auto Event1 = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) { B[it] = A[it] * 2; });
  });

  // event::wait() creates a host_sync node (partition boundary)
  Event1.wait();

  // Part 2: operations after the event wait
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) { C[it] = B[it] + A[it]; });
  });

  Graph.end_recording();

  auto ExecGraph = Graph.finalize();
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
  Queue.wait_and_throw();

  // Verify results
  std::vector<int> OutputB(N), OutputC(N);
  Queue.memcpy(OutputB.data(), B, N * sizeof(int)).wait();
  Queue.memcpy(OutputC.data(), C, N * sizeof(int)).wait();

  for (size_t i = 0; i < N; i++) {
    int a = static_cast<int>(i);
    int expected_b = a * 2;
    int expected_c = expected_b + a;

    assert(check_value(i, expected_b, OutputB[i], "B"));
    assert(check_value(i, expected_c, OutputC[i], "C"));
  }

  // Re-execute with different input
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      A[it] = static_cast<int>(it) + 5;
      B[it] = 0;
      C[it] = 0;
    });
  }).wait();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
  Queue.wait_and_throw();

  Queue.memcpy(OutputB.data(), B, N * sizeof(int)).wait();
  Queue.memcpy(OutputC.data(), C, N * sizeof(int)).wait();

  for (size_t i = 0; i < N; i++) {
    int a = static_cast<int>(i) + 5;
    int expected_b = a * 2;
    int expected_c = expected_b + a;

    assert(check_value(i, expected_b, OutputB[i], "B (second execution)"));
    assert(check_value(i, expected_c, OutputC[i], "C (second execution)"));
  }

  sycl::free(A, Queue);
  sycl::free(B, Queue);
  sycl::free(C, Queue);

  return 0;
}
