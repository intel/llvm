// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using ZE_DEBUG
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests memcpy operation using device USM and an in-order queue.

#include "../graph_common.hpp"

int main() {
  property_list properties{property::queue::in_order()};
  queue Queue{properties};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  float *X = malloc_device<float>(N, Queue);
  float *Y = malloc_device<float>(N, Queue);
  float *Z = malloc_device<float>(N, Queue);

  // Shouldn't be captured in graph as a dependency
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      X[it] = 0.0f;
      Y[it] = 0.0f;
      Z[it] = 0.0f;
    });
  });

  Graph.begin_recording(Queue);

  auto InitEvent = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      X[it] = 1.0f;
      Y[it] = 2.0f;
      Z[it] = 3.0f;
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
  Queue.submit([&](handler &CGH) { CGH.memcpy(Y, X, N * sizeof(float)); });

  // Double Y to 2.0
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> it) { Y[it] *= 2.0f; });
  });

  // memcpy from 2.0 Y values to Z
  Queue.submit([&](handler &CGH) { CGH.memcpy(Z, Y, N * sizeof(float)); });

  Graph.end_recording();

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });

  std::vector<float> Output(N);
  Queue.memcpy(Output.data(), Z, N * sizeof(float)).wait();

  for (size_t i = 0; i < N; i++) {
    assert(Output[i] == 2.0f);
  }

  sycl::free(X, Queue);
  sycl::free(Y, Queue);
  sycl::free(Z, Queue);

  return 0;
}
