// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests using a dynamic command-group object with dynamic parameters of
// different types

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

  int ScalarValue = 17;
  exp_ext::dynamic_parameter DynParamScalar(Graph, ScalarValue);
  exp_ext::dynamic_parameter DynParamPtr(Graph, PtrA);

  // Kernel has 2 dynamic parameters, one of scalar type & one of ptr type
  auto CGFA = [&](handler &CGH) {
    CGH.set_arg(0, DynParamPtr);
    CGH.set_arg(1, DynParamScalar);
    // TODO: Use the free function kernel extension instead of regular kernels
    // when available.
    CGH.single_task([=]() {
      for (size_t i = 0; i < Size; i++) {
        PtrA[i] = ScalarValue;
      }
    });
  };

  // Kernel has a single argument, a dynamic parameter of ptr type
  auto CGFB = [&](handler &CGH) {
    CGH.set_arg(0, DynParamPtr);
    // TODO: Use the free function kernel extension instead of regular kernels
    // when available.
    CGH.single_task([=]() {
      for (size_t i = 0; i < Size; i++) {
        PtrA[i] = ScalarValue;
      }
    });
  };

  // Kernel has a two arguments, an immutable ptr type argument and a
  // dynamic parameter of scalar type.
  auto CGFC = [&](handler &CGH) {
    CGH.set_arg(1, DynParamScalar);
    // TODO: Use the free function kernel extension instead of regular kernels
    // when available.
    CGH.single_task([=]() {
      for (size_t i = 0; i < Size; i++) {
        PtrC[i] = ScalarValue;
      }
    });
  };

  // Kernel has a single argument, of immutable pointer type
  auto CGFD = [&](handler &CGH) {
    // TODO: Use the free function kernel extension instead of regular kernels
    // when available.
    CGH.single_task([=]() {
      for (size_t i = 0; i < Size; i++) {
        PtrA[i] = ScalarValue;
      }
    });
  };

  auto DynamicCG =
      exp_ext::dynamic_command_group(Graph, {CGFA, CGFB, CGFC, CGFD});
  auto DynamicCGNode = Graph.add(DynamicCG);

  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  auto ExecuteGraphAndVerifyResults = [&](int A, int B, int C) {
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
      assert(HostDataA[i] == A);
      assert(HostDataB[i] == B);
      assert(HostDataC[i] == C);
    }
  };
  // CGFA using PtrA and ScalarValue in its dynamic parameters
  ExecuteGraphAndVerifyResults(ScalarValue, 0, 0);

  // CGFA using PtrB and UpdatedScalarValue in its dynamic parameters
  DynParamPtr.update(PtrB);
  int UpdatedScalarValue = 42;
  DynParamScalar.update(UpdatedScalarValue);
  ExecGraph.update(DynamicCGNode);
  ExecuteGraphAndVerifyResults(0, UpdatedScalarValue, 0);

  // CGFB using PtrB in its dynamic parameter and immutable ScalarValue
  DynamicCG.set_active_cgf(1);
  ExecGraph.update(DynamicCGNode);
  ExecuteGraphAndVerifyResults(0, ScalarValue, false);

  // CGFC using immutable PtrC and UpdatedScalarValue in its dynamic parameter
  DynamicCG.set_active_cgf(2);
  ExecGraph.update(DynamicCGNode);
  ExecuteGraphAndVerifyResults(0, 0, UpdatedScalarValue);

  // CGFD using immutable PtrA and immutable ScalarValue for arguments
  DynamicCG.set_active_cgf(3);
  ExecGraph.update(DynamicCGNode);
  ExecuteGraphAndVerifyResults(ScalarValue, 0, 0);

  sycl::free(PtrA, Queue);
  sycl::free(PtrB, Queue);
  sycl::free(PtrC, Queue);

  return 0;
}
