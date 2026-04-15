// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test that update() throws when called on a graph in native recording mode

#include "../../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue{property::queue::in_order{}};

  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};

  const size_t N = 1024;
  int *Data = malloc_device<int>(N, Queue);

  Graph.begin_recording(Queue);
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) { Data[idx] = idx; });
  });
  Graph.end_recording();

  auto ExecGraph = Graph.finalize({exp_ext::property::graph::updatable{}});

  if (!expectException([&]() { ExecGraph.update(Graph); },
                       "update(graph) with native recording enabled",
                       sycl::errc::feature_not_supported)) {
    free(Data, Queue);
    return 1;
  }

  if (!expectException(
          [&]() { ExecGraph.update(std::vector<exp_ext::node>{}); },
          "update(nodes) with native recording enabled",
          sycl::errc::feature_not_supported)) {
    free(Data, Queue);
    return 1;
  }

  free(Data, Queue);
  return 0;
}
