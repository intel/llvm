// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests that an event returned from adding a graph node using the queue
// recording API can be passed to `handler::depends_on` inside a node
// added using the explicit API. This should create a graph edge.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"

int main() {

  queue Queue{gpu_selector_v};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  float *Arr = malloc_device<float>(N, Queue);

  Graph.begin_recording(Queue);
  // `Event` corresponds to a graph node
  event Event = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) { Arr[idx] = 42.0f; });
  });
  Graph.end_recording(Queue);

  Graph.add([&](handler &CGH) {
    CGH.depends_on(Event); // creates edge to recorded graph node
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) { Arr[idx] *= 2.0f; });
  });

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  constexpr float ref = 42.0f * 2.0f;
  std::vector<float> Output(N);
  Queue.memcpy(Output.data(), Arr, N * sizeof(float)).wait();
  for (int i = 0; i < N; i++)
    assert(Output[i] == ref);

  sycl::free(Arr, Queue);

  return 0;
}
