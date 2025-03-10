// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//
// XFAIL: cuda
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/16004

// Tests updating dynamic_work_group_memory with a new size.

#include <sycl/group_barrier.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
namespace exp_ext = sycl::ext::oneapi::experimental;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((exp_ext::nd_range_kernel<1>))
void ff_1(exp_ext::dynamic_work_group_memory<int[]> DynLocalMem, int *PtrA) {
  const auto Item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
  size_t GlobalID = Item.get_global_id();
  auto LocalRange = Item.get_local_range(0);
  auto LocalMem = DynLocalMem.get();

  LocalMem[Item.get_local_id()] = LocalRange;
  group_barrier(Item.get_group());

  for (size_t i{0}; i < LocalRange; ++i) {
    PtrA[GlobalID] += LocalMem[i];
  }
}

int main() {
  constexpr int Size{1024};
  constexpr int LocalSize{32};
  nd_range<1> NDRange{Size, LocalSize};

  queue Queue{};
  context Ctxt{Queue.get_context()};

  int *PtrA = malloc_device<int>(Size, Queue);
  std::vector<int> HostDataA(Size);

  exp_ext::command_graph Graph{Ctxt, Queue.get_device()};
  exp_ext::dynamic_work_group_memory<int[]> DynLocalMem{Graph, LocalSize};

  Queue.memset(PtrA, 0, Size * sizeof(int)).wait();

#ifndef __SYCL_DEVICE_ONLY__
  kernel_bundle Bundle = get_kernel_bundle<bundle_state::executable>(Ctxt);
  kernel_id Kernel_id = exp_ext::get_kernel_id<ff_1>();
  kernel Kernel = Bundle.get_kernel(Kernel_id);
  auto KernelNode = Graph.add([&](handler &cgh) {
    cgh.set_arg(0, DynLocalMem);
    cgh.set_arg(1, PtrA);
    cgh.parallel_for(NDRange, Kernel);
  });

  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(PtrA, HostDataA.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostDataA[i] == LocalSize * LocalSize);
  }
  constexpr int NewLocalSize{64};

  Queue.memset(PtrA, 0, Size * sizeof(int)).wait();

  DynLocalMem.update(NewLocalSize);
  KernelNode.update_nd_range(nd_range<1>{range{Size}, range{NewLocalSize}});
  ExecGraph.update(KernelNode);

  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(PtrA, HostDataA.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostDataA[i] == NewLocalSize * NewLocalSize);
  }

#endif
  free(PtrA, Queue);
  return 0;
}
