// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using ZE_DEBUG
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Test executing a graph created with the explicit API multiple times.

#include "../graph_common.hpp"

int main() {

  queue Queue{gpu_selector_v};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  float *Arr = malloc_device<float>(N, Queue);
  float ZeroPattern = 0.0f;
  Queue.fill(Arr, ZeroPattern, N).wait();

  Graph.add([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Arr[i] += 1;
    });
  });

  std::vector<float> Output(N);
  Queue.memcpy(Output.data(), Arr, N * sizeof(float)).wait();
  for (int i = 0; i < N; i++)
    assert(Output[i] == 0);

  auto ExecGraph = Graph.finalize();

  Queue.memcpy(Output.data(), Arr, N * sizeof(float)).wait();
  for (int i = 0; i < N; i++)
    assert(Output[i] == 0);

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  Queue.memcpy(Output.data(), Arr, N * sizeof(float)).wait();
  for (int i = 0; i < N; i++)
    assert(Output[i] == 1);

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  Queue.memcpy(Output.data(), Arr, N * sizeof(float)).wait();
  for (int i = 0; i < N; i++)
    assert(Output[i] == 2);

  sycl::free(Arr, Queue);

  return 0;
}
