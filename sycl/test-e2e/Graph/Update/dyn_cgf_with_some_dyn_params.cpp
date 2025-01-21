// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests using a dynamic command-group object where some but not all the
// command-groups use dynamic parameters.

#include "../graph_common.hpp"

int main() {
  queue Queue{};
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  int *PtrA = malloc_device<int>(Size, Queue);
  int *PtrB = malloc_device<int>(Size, Queue);
  int *PtrC = malloc_device<int>(Size, Queue);

  std::vector<int> HostDataA(Size);
  std::vector<int> HostDataB(Size);
  std::vector<int> HostDataC(Size);

  exp_ext::dynamic_parameter DynParam(Graph, PtrA);

  auto CGFA = [&](handler &CGH) {
    CGH.set_arg(0, DynParam);
    // TODO: Use the free function kernel extension instead of regular kernels
    // when available.
    CGH.single_task([=]() {
      for (size_t i = 0; i < Size; i++) {
        PtrA[i] = i;
      }
    });
  };

  auto CGFB = [&](handler &CGH) {
    CGH.set_arg(0, DynParam);
    // TODO: Use the free function kernel extension instead of regular kernels
    // when available.
    CGH.single_task([=]() {
      for (size_t i = 0; i < Size; i++) {
        PtrA[i] = i;
      }
    });
  };

  auto CGFC = [&](handler &CGH) {
    CGH.single_task([=]() {
      for (size_t i = 0; i < Size; i++) {
        PtrC[i] = i;
      }
    });
  };

  auto DynamicCG = exp_ext::dynamic_command_group(Graph, {CGFA, CGFB, CGFC});
  auto DynamicCGNode = Graph.add(DynamicCG);
  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  auto ExecuteGraphAndVerifyResults = [&](bool A, bool B, bool C) {
    Queue.memset(PtrA, 0, Size * sizeof(int));
    Queue.memset(PtrB, 0, Size * sizeof(int));
    Queue.memset(PtrC, 0, Size * sizeof(int));
    Queue.wait();

    Queue.ext_oneapi_graph(ExecGraph).wait();

    Queue.copy(PtrA, HostDataA.data(), Size);
    Queue.copy(PtrB, HostDataB.data(), Size);
    Queue.copy(PtrC, HostDataC.data(), Size);
    Queue.wait();

    for (size_t i = 0; i < Size; i++) {
      assert(HostDataA[i] == (A ? i : 0));
      assert(HostDataB[i] == (B ? i : 0));
      assert(HostDataC[i] == (C ? i : 0));
    }
  };
  // CGFA with DynParam using PtrA
  ExecuteGraphAndVerifyResults(true, false, false);

  // CGFA with DynParam using PtrB
  DynParam.update(PtrB);
  ExecGraph.update(DynamicCGNode);
  ExecuteGraphAndVerifyResults(false, true, false);

  // CGFB with DynParam using PtrB
  DynamicCG.set_active_index(1);
  ExecGraph.update(DynamicCGNode);
  ExecuteGraphAndVerifyResults(false, true, false);

  // CGFC unconditionally using PtrC
  DynamicCG.set_active_index(2);
  ExecGraph.update(DynamicCGNode);
  ExecuteGraphAndVerifyResults(false, false, true);

  sycl::free(PtrA, Queue);
  sycl::free(PtrB, Queue);
  sycl::free(PtrC, Queue);

  return 0;
}
