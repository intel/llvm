// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero && linux %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//
// UNSUPPORTED: opencl, level_zero

// Tests updating a graph node accessor argument using index-based explicit
// update while also submitting work using those accessors to a normal queue

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  const size_t N = 1024;

  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::assume_buffer_outlives_graph{}}};
  std::vector<int> HostDataA(N, 0);
  std::vector<int> HostDataB(N, 0);

  buffer BufA{HostDataA};
  buffer BufB{HostDataB};
  BufA.set_write_back(false);
  BufB.set_write_back(false);
  // Initial accessor for use in kernel and dynamic parameter
  auto AccA = BufA.get_access();
  auto AccB = BufB.get_access();
  exp_ext::dynamic_parameter InputParam(Graph, AccA);

  auto KernelNode = Graph.add([&](handler &cgh) {
    cgh.require(InputParam);
    cgh.set_arg(0, InputParam);
    // TODO: Use the free function kernel extension instead of regular kernels
    // when available.
    cgh.single_task([=]() {
      for (size_t i = 0; i < N; i++) {
        AccA[i] = i;
      }
    });
  });

  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  // Modify A before the graph executes
  Queue.submit([&](handler &cgh) {
    cgh.require(AccA);
    cgh.single_task([=]() {
      for (size_t i = 0; i < N; i++) {
        // Write a different value than above, this should be overwritten when
        // the graph executes.
        AccA[i] = i * 3;
      }
    });
  });

  // BufA should be filled with values
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(BufA.get_access(), HostDataA.data()).wait();
  Queue.copy(BufB.get_access(), HostDataB.data()).wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostDataA[i] == i);
    assert(HostDataB[i] == 0);
  }

  // Swap BufB to be the input
  InputParam.update(BufB.get_access());
  ExecGraph.update(KernelNode);

  // Modify B before the graph executes
  Queue.submit([&](handler &cgh) {
    cgh.require(AccB);
    cgh.single_task([=]() {
      for (size_t i = 0; i < N; i++) {
        // Write a different value than above, this should be overwritten when
        // the graph executes.
        AccB[i] = i * 3;
      }
    });
  });

  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(BufA.get_access(), HostDataA.data()).wait();
  Queue.copy(BufB.get_access(), HostDataB.data()).wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostDataA[i] == i);
    assert(HostDataB[i] == i);
  }
  return 0;
}
