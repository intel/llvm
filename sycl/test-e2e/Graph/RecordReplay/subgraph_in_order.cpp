// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK


// Tests adding a sub-graph to an in-order queue.

#include "../graph_common.hpp"

int main() {
  property_list properties{property::queue::in_order()};
  queue Queue{gpu_selector_v, properties};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph SubGraph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  float *X = malloc_device<float>(N, Queue);

  SubGraph.begin_recording(Queue);

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      const size_t i = it[0];
      X[i] *= 2.0f;
    });
  });

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      const size_t i = it[0];
      X[i] += 0.5f;
    });
  });

  SubGraph.end_recording(Queue);

  auto ExecSubGraph = SubGraph.finalize();

  Graph.begin_recording(Queue);

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      const size_t i = it[0];
      X[i] = 1.0f;
    });
  });

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecSubGraph); });

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> it) {
      const size_t i = it[0];
      X[i] += 3.0f;
    });
  });

  Graph.end_recording();

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });

  float Output;
  Queue.memcpy(&Output, X, sizeof(float)).wait();

  assert(Output == 5.5f);

  sycl::free(X, Queue);

  return 0;
}
