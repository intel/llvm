// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using ZE_DEBUG
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Tests that a command-group function can capture variables by reference
// and still work correctly as a graph node.

#include "../graph_common.hpp"

const size_t N = 10;
const float ExpectedValue = 42.0f;

void run_some_kernel(queue Queue, float *Data) {
  // 'Data' is captured by ref here but will have gone out of scope when the
  // CGF is later run when the graph is executed.
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Data[i] = ExpectedValue;
    });
  });
}

int main() {

  queue Queue{default_selector_v};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  float *Arr = malloc_device<float>(N, Queue);

  Graph.begin_recording(Queue);
  run_some_kernel(Queue, Arr);
  Graph.end_recording(Queue);

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  std::vector<float> Output(N);
  Queue.memcpy(Output.data(), Arr, N * sizeof(float)).wait();
  for (size_t i = 0; i < N; i++) {
    assert(Output[i] == ExpectedValue);
  }

  sycl::free(Arr, Queue);

  return 0;
}
