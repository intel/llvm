// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//
// XFAIL: cuda
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/16004

// Tests updating multiple parameters to a singlegraph node using index-based
// explicit update

#include "../../graph_common.hpp"
#include "free_function_kernels.hpp"

int main() {
  queue Queue{};
  context ctxt{Queue.get_context()};

  exp_ext::command_graph Graph{ctxt, Queue.get_device()};

  int *PtrA = malloc_device<int>(Size, Queue);
  int *PtrB = malloc_device<int>(Size, Queue);
  int *PtrC = malloc_device<int>(Size, Queue);

  std::vector<int> HostDataA(Size);
  std::vector<int> HostDataB(Size);
  std::vector<int> HostDataC(Size);
  std::vector<int> OutData(Size);

  std::iota(HostDataA.begin(), HostDataA.end(), 10);
  std::iota(HostDataB.begin(), HostDataB.end(), 100);

  Queue.memcpy(PtrA, HostDataA.data(), Size * sizeof(int)).wait();
  Queue.memcpy(PtrB, HostDataB.data(), Size * sizeof(int)).wait();
  Queue.memset(PtrC, 0, Size * sizeof(int)).wait();

  exp_ext::dynamic_parameter ParamA(Graph, PtrA);
  exp_ext::dynamic_parameter ParamB(Graph, PtrB);
  exp_ext::dynamic_parameter ParamOut(Graph, PtrC);

  nd_range<1> NDRange{Size, 32};

#ifndef __SYCL_DEVICE_ONLY__
  kernel_bundle Bundle = get_kernel_bundle<bundle_state::executable>(ctxt);
  kernel_id Kernel_id = exp_ext::get_kernel_id<ff_5>();
  kernel Kernel = Bundle.get_kernel(Kernel_id);
  auto KernelNode = Graph.add([&](handler &cgh) {
    cgh.set_arg(0, ParamA);
    cgh.set_arg(1, ParamB);
    cgh.set_arg(2, ParamOut);
    cgh.parallel_for(NDRange, Kernel);
  });

  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  // PtrA should be filled with values
  Queue.ext_oneapi_graph(ExecGraph).wait();

  // Copy to output data to preserve original data for verifying += op
  Queue.copy(PtrC, OutData.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(OutData[i] == HostDataC[i] + (HostDataA[i] * HostDataB[i]));
  }

  // Update C's host data
  HostDataC = OutData;

  // Swap PtrB to be the input
  ParamOut.update(PtrB);
  ParamB.update(PtrC);

  ExecGraph.update(KernelNode);
  Queue.ext_oneapi_graph(ExecGraph).wait();

  // Copy to output data to preserve original data for verifying += op
  Queue.copy(PtrB, OutData.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(OutData[i] == HostDataB[i] + (HostDataA[i] * HostDataC[i]));
  }
  sycl::free(PtrA, Queue);
  sycl::free(PtrB, Queue);
  sycl::free(PtrC, Queue);
#endif
  return 0;
}
