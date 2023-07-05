// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests the explicit API interface for adding empty nodes
// and adding dependencies (edges) on those empty nodes, and that
// no_cycle_check is accepted as a command_graph construction property.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"

int main() {
  queue Queue{gpu_selector_v};

  auto MyProperties = property_list{exp_ext::property::graph::no_cycle_check()};
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device(),
                               MyProperties};

  const size_t N = 10;
  float *Arr = malloc_device<float>(N, Queue);

  auto Start = Graph.add();

  auto Init = Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
          size_t i = idx;
          Arr[i] = 0;
        });
      },
      {exp_ext::property::node::depends_on(Start)});

  auto Empty = Graph.add({});
  auto Empty2 = Graph.add({});

  Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
          size_t i = idx;
          Arr[i] = 1;
        });
      },
      {exp_ext::property::node::depends_on(Init, Empty, Empty2)});

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  std::vector<float> HostData(N);
  Queue.memcpy(HostData.data(), Arr, N * sizeof(float)).wait();

  for (int i = 0; i < N; i++)
    assert(HostData[i] == 1.f);

  sycl::free(Arr, Queue);

  return 0;
}
