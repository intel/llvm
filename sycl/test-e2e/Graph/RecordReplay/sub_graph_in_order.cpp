// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//

// Tests adding a sub-graph to an in-order queue.

#include "../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  property_list Properties{property::queue::in_order{}};
  queue Queue{Properties};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph SubGraph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  int *X = malloc_device<int>(N, Queue);

  SubGraph.begin_recording(Queue);

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) { X[it] *= 2; });
  });

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) { X[it] += 1; });
  });

  SubGraph.end_recording(Queue);

  auto ExecSubGraph = SubGraph.finalize();

  Graph.begin_recording(Queue);

  Queue.submit(
      [&](handler &CGH) { CGH.parallel_for(N, [=](id<1> it) { X[it] = 1; }); });

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecSubGraph); });

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> it) { X[it] += 3; });
  });

  Graph.end_recording();

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });

  int Output;
  Queue.memcpy(&Output, X, sizeof(int)).wait();

  assert(Output == 6);

  sycl::free(X, Queue);

  return 0;
}
