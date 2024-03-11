// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero && linux %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//

// Tests that a command-group function can capture variables by reference
// and still work correctly as a graph node.

#include "../graph_common.hpp"

const size_t N = 10;
const int ExpectedValue = 42;

void run_some_kernel(queue Queue, int *Data) {
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

  int *Arr = malloc_device<int>(N, Queue);

  Graph.begin_recording(Queue);
  run_some_kernel(Queue, Arr);
  Graph.end_recording(Queue);

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  std::vector<int> Output(N);
  Queue.memcpy(Output.data(), Arr, N * sizeof(int)).wait();
  for (size_t i = 0; i < N; i++) {
    assert(check_value(i, ExpectedValue, Output[i], "Output"));
  }

  sycl::free(Arr, Queue);

  return 0;
}
