// XFAIL: run-mode && linux && arch-intel_gpu_bmg_g21 && spirv-backend
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/19586
// RUN: %{build} -Wno-error=deprecated-declarations -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests updating a graph node scalar argument using index-based explicit update

#include "../../graph_common.hpp"
#include "free_function_kernels.hpp"

int main() {
  queue Queue{};
  context Ctxt{Queue.get_context()};

  exp_ext::command_graph Graph{Ctxt, Queue.get_device()};

  int *DeviceData = malloc_device<int>(Size, Queue);

  int ScalarValue = 17;

  std::vector<int> HostData(Size);

  Queue.memset(DeviceData, 0, Size * sizeof(int)).wait();

  exp_ext::dynamic_parameter InputParam(Graph, ScalarValue);

  kernel_bundle Bundle = get_kernel_bundle<bundle_state::executable>(Ctxt);
  kernel_id Kernel_id = exp_ext::get_kernel_id<ff_6>();
  kernel Kernel = Bundle.get_kernel(Kernel_id);
  auto KernelNode = Graph.add([&](handler &cgh) {
    cgh.set_arg(0, DeviceData);
    cgh.set_arg(1, InputParam);
    cgh.single_task(Kernel);
  });

  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  // DeviceData should be filled with current ScalarValue (17)
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(DeviceData, HostData.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostData[i] == 17);
  }

  // Update ScalarValue to be 99 instead
  InputParam.update(99);
  ExecGraph.update(KernelNode);
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(DeviceData, HostData.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostData[i] == 99);
  }
  sycl::free(DeviceData, Queue);

  return 0;
}
