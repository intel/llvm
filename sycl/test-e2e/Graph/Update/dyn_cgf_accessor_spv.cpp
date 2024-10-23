// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %S/../Inputs/Kernels/update_with_indices_accessor.spv
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out %S/../Inputs/Kernels/update_with_indices_accessor.spv 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out %S/../Inputs/Kernels/update_with_indices_accessor.spv 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// REQUIRES: level_zero
// XFAIL: level_zero
// XFAIL-TRACKER: OFNAAO-307

// Tests updating an accessor argument to a graph node created from SPIR-V
// using dynamic command-groups.

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

  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

  buffer<int> BufA{sycl::range<1>(Size)};
  buffer<int> BufB{sycl::range<1>(Size)};
  BufA.set_write_back(false);
  BufB.set_write_back(false);

  auto AccA = BufA.get_access();
  auto AccB = BufB.get_access();

  auto CGFA = [&](handler &CGH) {
    CGH.require(AccA);
    CGH.set_arg(0, AccA);
    CGH.single_task(kernel);
  };

  auto CGFB = [&](handler &CGH) {
    CGH.require(AccB);
    CGH.set_arg(0, AccB);
    CGH.single_task(kernel);
  };

  auto DynamicCG = exp_ext::dynamic_command_group(Queue, {CGFA, CGFB});
  auto DynamicCGNode = Graph.add(DynamicCG);
  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  Queue.ext_oneapi_graph(ExecGraph).wait();

  std::vector<int> HostDataA(Size, 0);
  std::vector<int> HostDataB(Size, 0);
  Queue.copy(BufA.get_access(), HostDataA.data()).wait();
  Queue.copy(BufB.get_access(), HostDataB.data()).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostDataA[i] == i);
    assert(HostDataB[i] == 0);
  }

  DynamicCG.set_active_cgf(1);
  ExecGraph.update(DynamicCGNode);

  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(BufA.get_access(), HostDataA.data()).wait();
  Queue.copy(BufB.get_access(), HostDataB.data()).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostDataA[i] == i);
    assert(HostDataB[i] == i);
  }
  return 0;
}
