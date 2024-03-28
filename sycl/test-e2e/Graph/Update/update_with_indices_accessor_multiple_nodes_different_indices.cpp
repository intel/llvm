// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero && linux %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//
// UNSUPPORTED: opencl, level_zero

// Tests updating a single dynamic parameter which is registered with multiple
// graph nodes where it has a different argument index in each node

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

  auto KernelNodeA = Graph.add([&](handler &cgh) {
    cgh.require(AccB);
    cgh.require(InputParam);
    // Arg index is 4 here
    cgh.set_arg(4, InputParam);
    // TODO: Use the free function kernel extension instead of regular kernels
    // when available.
    cgh.single_task([=]() {
      for (size_t i = 0; i < N; i++) {
        AccB[i] = 0;
        AccA[i] = i;
      }
    });
  });

  auto KernelNodeB = Graph.add(
      [&](handler &cgh) {
        cgh.require(InputParam);
        // Arg index is 0 here
        cgh.set_arg(0, InputParam);
        // TODO: Use the free function kernel extension instead of regular
        // kernels when available.
        cgh.single_task([=]() {
          for (size_t i = 0; i < N; i++) {
            AccA[i] += i;
          }
        });
      },
      exp_ext::property::node::depends_on{KernelNodeA});

  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  // AccA should be filled with values
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(AccA, HostDataA.data()).wait();
  Queue.copy(AccB, HostDataB.data()).wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostDataA[i] == i * 2);
    assert(HostDataB[i] == 0);
  }

  // Swap AccB to be the input
  InputParam.update(AccB);
  ExecGraph.update({KernelNodeA, KernelNodeB});
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(AccA, HostDataA.data()).wait();
  Queue.copy(AccB, HostDataB.data()).wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostDataA[i] == i * 2);
    assert(HostDataB[i] == i * 2);
  }
  return 0;
}
