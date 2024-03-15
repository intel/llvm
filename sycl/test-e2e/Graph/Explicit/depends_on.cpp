// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero && linux %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//

// Tests that an event returned from adding a graph node using the queue
// recording API can be passed to `handler::depends_on` inside a node
// added using the explicit API. This should create a graph edge.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  int *Arr = malloc_device<int>(N, Queue);

  Graph.begin_recording(Queue);
  // `Event` corresponds to a graph node
  event Event = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) { Arr[idx] = 42; });
  });
  Graph.end_recording(Queue);

  Graph.add([&](handler &CGH) {
    CGH.depends_on(Event); // creates edge to recorded graph node
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) { Arr[idx] *= 2; });
  });

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  constexpr int Ref = 42 * 2;
  std::vector<int> Output(N);
  Queue.memcpy(Output.data(), Arr, N * sizeof(int)).wait();
  for (size_t i = 0; i < N; i++)
    assert(check_value(i, Ref, Output[i], "Output"));

  sycl::free(Arr, Queue);

  return 0;
}
