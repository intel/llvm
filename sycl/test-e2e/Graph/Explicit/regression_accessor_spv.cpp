// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %S/../Inputs/Kernels/update_with_indices_accessor.spv
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out %S/../Inputs/Kernels/update_with_indices_accessor.spv 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out %S/../Inputs/Kernels/update_with_indices_accessor.spv 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// REQUIRES: level_zero

// Modified version of Update/update_with_indices_accessor_spv.cpp that does
// not require the full graph aspect, test was hanging after some changes to
// kernel bundles so adding this test for the CI which doesn't support update

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

  std::vector<int> HostDataA(N, 0);

  buffer BufA{HostDataA};
  BufA.set_write_back(false);

  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    Graph.add([&](handler &cgh) {
      auto Acc = BufA.get_access(cgh);

      cgh.set_arg(0, Acc);
      cgh.single_task(kernel);
    });

    auto ExecGraph = Graph.finalize();
    Queue.ext_oneapi_graph(ExecGraph).wait();
  }
  // Copy data back to host
  Queue.copy(BufA.get_access(), HostDataA.data()).wait();

  for (size_t i = 0; i < N; i++) {
    assert(HostDataA[i] == i);
  }
  return 0;
}
