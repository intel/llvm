// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests that external event dependencies (events from before graph recording)
// are handled as partition boundaries when allow_wait_recording is set.
// This covers the case where a command submitted during recording depends on
// an event created outside the recording scope

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
    CGH.parallel_for(N, [=](id<1> it) { A[it] = static_cast<int>(it); });
  });

  auto ExternalEvent = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) { B[it] = A[it] * 2; });
  });

  Graph.begin_recording(Queue);

  // Submit a kernel that depends on the pre-recording event.
  // Without allow_wait_recording this would throw:
  //   "Graph nodes cannot depend on events from outside the graph."
  Queue.submit([&](handler &CGH) {
    CGH.depends_on(ExternalEvent);
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

  // Re-execute with different input to verify the graph works multiple times
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      A[it] = static_cast<int>(it) + 10;
      B[it] = A[it] * 2;
      C[it] = 0;
    });
  }).wait();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
  Queue.wait_and_throw();

  Queue.memcpy(OutputC.data(), C, N * sizeof(int)).wait();

  for (size_t i = 0; i < N; i++) {
    int a = static_cast<int>(i) + 10;
    int b = a * 2;
    int expected_c = b + a;

    assert(check_value(i, expected_c, OutputC[i], "C (second execution)"));
  }

  sycl::free(A, Queue);
  sycl::free(B, Queue);
  sycl::free(C, Queue);

  return 0;
}
