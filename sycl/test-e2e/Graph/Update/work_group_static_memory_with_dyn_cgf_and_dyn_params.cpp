// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//
// sycl_ext_oneapi_work_group_static is not supported on AMD
// UNSUPPORTED: hip

// Tests using sycl_ext_oneapi_work_group_static in a graph node with dynamic
// cgf and dynamic parameter

#include "../graph_common.hpp"
#include <sycl/ext/oneapi/work_group_static.hpp>

constexpr size_t WgSize = 32;

// Local mem used in kernel
sycl::ext::oneapi::experimental::work_group_static<int[WgSize]> LocalIDBuff;

int main() {
  queue Queue;
  exp_ext::command_graph Graph{Queue};

  int *PtrA = malloc_device<int>(Size, Queue);
  int *PtrB = malloc_device<int>(Size, Queue);

  std::vector<int> HostDataA(Size);
  std::vector<int> HostDataB(Size);

  exp_ext::dynamic_parameter DynParam(Graph, PtrA);

  auto CGFA = [&](handler &CGH) {
    CGH.set_arg(0, DynParam);
    CGH.parallel_for(nd_range({Size}, {WgSize}), [=](nd_item<1> Item) {
      LocalIDBuff[Item.get_local_linear_id()] = Item.get_local_linear_id();

      Item.barrier();

      // Check that the memory is accessible from other work-items
      size_t LocalIdx = Item.get_local_linear_id() ^ 1;
      size_t GlobalIdx = Item.get_global_linear_id() ^ 1;
      PtrA[GlobalIdx] = LocalIDBuff[LocalIdx];
    });
  };

  auto CGFB = [&](handler &CGH) {
    CGH.set_arg(0, DynParam);
    CGH.parallel_for(nd_range({Size}, {WgSize}), [=](nd_item<1> Item) {
      LocalIDBuff[Item.get_local_linear_id()] = Item.get_local_linear_id();

      Item.barrier();

      // Check that the memory is accessible from other work-items
      size_t LocalIdx = Item.get_local_linear_id() ^ 1;
      size_t GlobalIdx = Item.get_global_linear_id() ^ 1;
      PtrA[GlobalIdx] = LocalIDBuff[LocalIdx] - 1;
    });
  };

  auto DynamicCG = exp_ext::dynamic_command_group(Graph, {CGFA, CGFB});
  auto DynamicCGNode = Graph.add(DynamicCG);
  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  auto ExecuteGraphAndVerifyResults = [&](bool A, bool B, bool nextCGF) {
    Queue.memset(PtrA, 0, Size * sizeof(int));
    Queue.memset(PtrB, 0, Size * sizeof(int));
    Queue.wait();

    Queue.ext_oneapi_graph(ExecGraph).wait();

    Queue.copy(PtrA, HostDataA.data(), Size);
    Queue.copy(PtrB, HostDataB.data(), Size);
    Queue.wait();

    for (size_t i = 0; i < Size; i++) {
      int Ref = nextCGF ? (i % WgSize) - 1 : i % WgSize;
      assert(HostDataA[i] == (A ? Ref : 0));
      assert(HostDataB[i] == (B ? Ref : 0));
    }
  };

  ExecuteGraphAndVerifyResults(true, false, false);

  DynParam.update(PtrB);
  ExecGraph.update(DynamicCGNode);
  ExecuteGraphAndVerifyResults(false, true, false);

  DynamicCG.set_active_index(1);
  ExecGraph.update(DynamicCGNode);
  ExecuteGraphAndVerifyResults(false, true, true);

  free(PtrA, Queue);
  free(PtrB, Queue);
  return 0;
}
