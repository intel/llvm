// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests that recording can be paused and resumed

#include "../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue{property::queue::in_order{}};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 1024;
  int *Data = malloc_device<int>(N, Queue);

  QueueStateVerifier Verifier(Queue);
  Verifier.verify(EXECUTING);

  Graph.begin_recording(Queue);
  Verifier.verify(RECORDING);

  Queue.parallel_for(range<1>{N},
                     [=](id<1> idx) { Data[idx] = static_cast<int>(idx); });

  // Pause recording
  Graph.end_recording(Queue);
  Verifier.verify(EXECUTING);

  // NOT recorded
  Queue.parallel_for(range<1>{N}, [=](id<1> idx) { Data[idx] += 1000; }).wait();

  // resume
  Graph.begin_recording(Queue);
  Verifier.verify(RECORDING);

  Queue.parallel_for(range<1>{N},
                     [=](id<1> idx) { Data[idx] = Data[idx] * 2; });

  Graph.end_recording(Queue);
  Verifier.verify(EXECUTING);

  // Reset Data to known state before executing the graph
  Queue.parallel_for(range<1>{N}, [=](id<1> idx) { Data[idx] = 0; }).wait();

  auto ExecGraph = Graph.finalize();

  Queue.ext_oneapi_graph(ExecGraph);
  Queue.wait();

  // Expected: don't run the eager +1000 kernel.
  std::vector<int> HostData(N);
  Queue.memcpy(HostData.data(), Data, N * sizeof(int)).wait();

  for (size_t i = 0; i < N; i++) {
    int Expected = static_cast<int>(i) * 2;
    assert(check_value(i, Expected, HostData[i], "Data"));
  }

  free(Data, Queue);
  return 0;
}
