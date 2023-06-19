// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// This test creates a graph with empty nodes.
// Empty nodes are used to enforce graph dependencies.
// This test ensures that empty nodes can depend on other nodes
// as well as other nodes can depend on empty nodes.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"

int main() {
  queue Queue;

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  float *Arr = malloc_device<float>(N, Queue);

  Graph.begin_recording(Queue);

  auto Init = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Arr[i] = static_cast<float>(i);
    });
  });

  auto Empty1 = Queue.submit([&](handler &) {});
  auto Empty2 = Queue.submit([&](handler &CGH) { CGH.depends_on({Init}); });

  Queue.submit([&](handler &CGH) {
    CGH.depends_on({Empty1, Empty2});
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Arr[i] += 1.0f;
    });
  });

  Graph.end_recording(Queue);

  auto ExecGraph = Graph.finalize();
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  std::vector<float> HostData(N);
  Queue.memcpy(HostData.data(), Arr, N * sizeof(float)).wait();
  for (int i = 0; i < N; i++)
    assert(HostData[i] == static_cast<float>(i) + 1.0f);

  free(Arr, Queue);

  return 0;
}
