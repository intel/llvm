// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests updating a graph node using a USM pointer set to nullptr

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  const size_t N = 1024;

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  int *PtrOut = malloc_device<int>(N, Queue);
  int *PtrIn = malloc_device<int>(N, Queue);

  Queue.memset(PtrOut, 0, N * sizeof(int)).wait();
  int PtrPattern = 42;
  Queue.fill(PtrIn, PtrPattern, N).wait();

  exp_ext::dynamic_parameter InputParam(Graph, PtrIn);
  int DefaultPattern = 10;
  auto KernelNode = Graph.add([&](handler &cgh) {
    cgh.set_arg(0, InputParam);
    cgh.set_arg(1, PtrOut);
    cgh.single_task([=]() {
      for (size_t i = 0; i < N; i++) {
        if (PtrIn) {
          PtrOut[i] = PtrIn[i];
        } else {
          PtrOut[i] = DefaultPattern;
        }
      }
    });
  });

  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  // PtrA should be filled with values
  Queue.ext_oneapi_graph(ExecGraph).wait();

  std::vector<int> HostData(N);
  Queue.copy(PtrOut, HostData.data(), N).wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostData[i] == PtrPattern);
  }

  // Swap Input to nullptr
  int *NullPtr = nullptr;
  InputParam.update(NullPtr);
  ExecGraph.update(KernelNode);
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(PtrOut, HostData.data(), N).wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostData[i] == DefaultPattern);
  }

  free(PtrIn, Queue);
  free(PtrOut, Queue);
  return 0;
}
