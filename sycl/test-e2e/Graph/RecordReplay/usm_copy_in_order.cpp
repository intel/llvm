// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//
//
// Intended - USM copy command not supported for OpenCL
// UNSUPPORTED: opencl

// Tests memcpy operation using device USM and an in-order queue.

#include "../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  property_list Properties{property::queue::in_order{}};
  queue Queue{Properties};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  int *X = malloc_device<int>(N, Queue);
  int *Y = malloc_device<int>(N, Queue);
  int *Z = malloc_device<int>(N, Queue);

  // Shouldn't be captured in graph as a dependency
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      X[it] = 0;
      Y[it] = 0;
      Z[it] = 0;
    });
  });

  Graph.begin_recording(Queue);

  auto InitEvent = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      X[it] = 1;
      Y[it] = 2;
      Z[it] = 3;
    });
  });
  Graph.end_recording(Queue);

  // Shouldn't be captured in graph as a dependency
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      X[it] += 0.5f;
      Y[it] += 0.5f;
      Z[it] += 0.5f;
    });
  });

  Graph.begin_recording(Queue);
  // memcpy 1 values from X to Y
  Queue.submit([&](handler &CGH) { CGH.memcpy(Y, X, N * sizeof(int)); });

  // Double Y to 2.0
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> it) { Y[it] *= 2; });
  });

  // memcpy from 2.0 Y values to Z
  Queue.submit([&](handler &CGH) { CGH.memcpy(Z, Y, N * sizeof(int)); });

  Graph.end_recording();

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });

  std::vector<int> Output(N);
  Queue.memcpy(Output.data(), Z, N * sizeof(int)).wait();

  const int Expected = 2;
  for (size_t i = 0; i < N; i++) {
    assert(check_value(i, Expected, Output[i], "Output"));
  }

  sycl::free(X, Queue);
  sycl::free(Y, Queue);
  sycl::free(Z, Queue);

  return 0;
}
