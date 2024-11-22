// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//
// The name mangling for free function kernels currently does not work with PTX.
// UNSUPPORTED: cuda

// Tests updating a graph node scalar argument using index-based explicit update

#include "../../graph_common.hpp"
#include "free_function_kernels.hpp"

int main() {
  queue Queue{};
  context ctxt{Queue.get_context()};

  const size_t N = 1024;

  exp_ext::command_graph Graph{ctxt, Queue.get_device()};

  int *DeviceData = malloc_device<int>(N, Queue);

  int ScalarValue = 17;

  std::vector<int> HostData(N);

  Queue.memset(DeviceData, 0, N * sizeof(int)).wait();

  exp_ext::dynamic_parameter InputParam(Graph, ScalarValue);

#ifndef __SYCL_DEVICE_ONLY__
  kernel_bundle Bundle = get_kernel_bundle<bundle_state::executable>(ctxt);
  kernel_id Kernel_id = exp_ext::get_kernel_id<ff_6>();
  kernel Kernel = Bundle.get_kernel(Kernel_id);
  auto KernelNode = Graph.add([&](handler &cgh) {
    cgh.set_arg(0, DeviceData);
    cgh.set_arg(1, InputParam);
    cgh.set_arg(2, N);
    cgh.single_task(Kernel);
  });

  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  // DeviceData should be filled with current ScalarValue (17)
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(DeviceData, HostData.data(), N).wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostData[i] == 17);
  }

  // Update ScalarValue to be 99 instead
  InputParam.update(99);
  ExecGraph.update(KernelNode);
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(DeviceData, HostData.data(), N).wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostData[i] == 99);
  }
#endif
  return 0;
}
