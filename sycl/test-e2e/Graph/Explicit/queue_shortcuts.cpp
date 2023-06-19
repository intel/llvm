// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests passing empty properties to graph APIs that take properties,
// as well as the queue shortcuts for submitting an executable graph.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"

int main() {

  queue Queue{gpu_selector_v};

  // Test passing empty property list, which is the default
  property_list EmptyProperties;
  exp_ext::command_graph Graph(Queue.get_context(), Queue.get_device(),
                               EmptyProperties);

  const size_t N = 10;
  float *Arr = malloc_device<float>(N, Queue);

  Graph.add([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Arr[i] = 1;
    });
  });

  auto ExecGraph = Graph.finalize(EmptyProperties);

  auto Event1 = Queue.ext_oneapi_graph(ExecGraph);
  auto Event2 = Queue.ext_oneapi_graph(ExecGraph, Event1);
  auto Event3 = Queue.ext_oneapi_graph(ExecGraph, Event1);
  Queue.ext_oneapi_graph(ExecGraph, {Event2, Event3}).wait();

  std::vector<float> Output(N);
  Queue.memcpy(Output.data(), Arr, N * sizeof(float)).wait();
  for (int i = 0; i < N; i++)
    assert(Output[i] == 1);

  sycl::free(Arr, Queue);

  return 0;
}
