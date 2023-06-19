// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// XFAIL:*
// Submit a graph as a subgraph more than once doesn't yet work.

// Tests creating a parent graph with the same sub-graph interleaved with
// other nodes.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"

int main() {
  queue Queue{gpu_selector_v};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph SubGraph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  float *X = malloc_device<float>(N, Queue);

  SubGraph.begin_recording(Queue);

  auto S1 = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      const size_t i = it[0];
      X[i] *= 2.0f;
    });
  });

  Queue.submit([&](handler &CGH) {
    CGH.depends_on(S1);
    CGH.parallel_for(N, [=](id<1> it) {
      const size_t i = it[0];
      X[i] += 0.5f;
    });
  });

  SubGraph.end_recording(Queue);

  auto ExecSubGraph = SubGraph.finalize();

  Graph.begin_recording(Queue);

  auto P1 = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      const size_t i = it[0];
      X[i] = 1.0f;
    });
  });

  auto P2 = Queue.submit([&](handler &CGH) {
    CGH.depends_on(P1);
    CGH.ext_oneapi_graph(ExecSubGraph);
  });

  auto P3 = Queue.submit([&](handler &CGH) {
    CGH.depends_on(P2);
    CGH.parallel_for(range<1>{N}, [=](id<1> it) {
      const size_t i = it[0];
      X[i] *= -1.0f;
    });
  });

  Queue.submit([&](handler &CGH) {
    CGH.depends_on(P3);
    CGH.ext_oneapi_graph(ExecSubGraph);
  });

  Graph.end_recording();

  auto ExecGraph = Graph.finalize();

  auto E = Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });

  std::vector<float> Output(N);
  Queue.memcpy(Output.data(), X, N * sizeof(float), E).wait();

  for (size_t i = 0; i < N; i++) {
    assert(Output[i] == -6.25f);
  }

  sycl::free(X, Queue);

  return 0;
}
