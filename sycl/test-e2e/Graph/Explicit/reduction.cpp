// REQUIRES: cuda || level_zero, gpu
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// Only run leak checks on Linux, as there is a known leak with reductions
// on Windows.
// RUN: %if linux && ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Verifies that the implementation changes to implement the graph
// extension don't regress reductions.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"

int main() {
  queue Queue{gpu_selector_v};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  float *Input = malloc_device<float>(N, Queue);
  float *Output = malloc_device<float>(1, Queue);

  auto Init = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      const size_t i = it[0];
      Input[i] = i;
    });
  });

  auto Event = Queue.submit([&](handler &CGH) {
    CGH.depends_on(Init);
    CGH.parallel_for(range<1>{N}, reduction(Output, 0.0f, std::plus()),
                     [=](id<1> idx, auto &Sum) { Sum += Input[idx]; });
  });

  auto ExecGraph = Graph.finalize();
  Queue.ext_oneapi_graph(ExecGraph).wait();

  float HostOut;
  Queue.memcpy(&HostOut, Output, sizeof(float)).wait();

  assert(HostOut == 45);

  sycl::free(Input, Queue);
  sycl::free(Output, Queue);

  return 0;
}
