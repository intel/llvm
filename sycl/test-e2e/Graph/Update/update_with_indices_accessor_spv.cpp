// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %S/../Inputs/Kernels/update_with_indices_accessor.spv
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out %S/../Inputs/Kernels/update_with_indices_accessor.spv 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out %S/../Inputs/Kernels/update_with_indices_accessor.spv 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// REQUIRES: level_zero

// Tests updating an accessor argument to a graph node created from SPIR-V
// using index-based explicit update

#include "../graph_common.hpp"

int main(int, char **argv) {
  queue Queue{};
  sycl::kernel_bundle KernelBundle = loadKernelsFromFile(Queue, argv[1]);
  const auto getKernel =
      [](sycl::kernel_bundle<sycl::bundle_state::executable> &bundle,
         const std::string &name) {
        return bundle.ext_oneapi_get_kernel(name);
      };

  kernel kernel = getKernel(
      KernelBundle, "_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE_");

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
  auto Acc = BufA.get_access();
  exp_ext::dynamic_parameter InputParam(Graph, Acc);

  auto KernelNode = Graph.add([&](handler &cgh) {
    cgh.require(InputParam);
    cgh.set_arg(0, InputParam);
    cgh.single_task(kernel);
  });

  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

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
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(BufA.get_access(), HostDataA.data()).wait();
  Queue.copy(BufB.get_access(), HostDataB.data()).wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostDataA[i] == i);
    assert(HostDataB[i] == i);
  }
  return 0;
}
