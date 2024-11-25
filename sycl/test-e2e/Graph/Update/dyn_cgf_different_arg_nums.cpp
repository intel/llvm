// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests updating a dynamic command-group with command-groups containing a
// different number of arguments.

#include "../graph_common.hpp"

int main() {
  queue Queue{};
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  int *Ptr = malloc_device<int>(Size, Queue);
  std::vector<int> HostData(Size);

  // 3 kernel arguments: Ptr, PatternA, PatternB
  int PatternA = 42;
  int PatternB = 0xA;
  auto CGFA = [&](handler &CGH) {
    CGH.parallel_for(
        Size, [=](item<1> Item) { Ptr[Item.get_id()] = PatternA + PatternB; });
  };

  // 2 kernel arguments: Ptr, MyPatternStruct
  struct PatternStruct {
    int PatternA;
    int PatternB;
  };
  PatternStruct MyPatternStruct{PatternA + 1, PatternB + 1};
  auto CGFB = [&](handler &CGH) {
    CGH.parallel_for(Size, [=](item<1> Item) {
      Ptr[Item.get_id()] = MyPatternStruct.PatternA + MyPatternStruct.PatternB;
    });
  };

  // 1 kernel argument: Ptr
  auto CGFC = [&](handler &CGH) {
    CGH.parallel_for(Size,
                     [=](item<1> Item) { Ptr[Item.get_id()] = 42 - 0xA; });
  };

  // 4 kernel argument: Ptr
  int PatternC = -12;
  auto CGFD = [&](handler &CGH) {
    CGH.parallel_for(Size, [=](item<1> Item) {
      Ptr[Item.get_id()] = PatternA + PatternB + PatternC;
    });
  };

  // CHECK: <--- urKernelSetArgPointer(
  // CHECK-SAME: .hKernel = [[KERNEL_HANDLE1:[0-9a-fA-Fx]+]]
  // CHECL-SAME: .argIndex = 0

  // CHECK:   <--- urKernelSetArgValue
  // CHECK-SAME: .hKernel = [[KERNEL_HANDLE1]]
  // CHECK-SAME: .argIndex = 1

  // CHECK:   <--- urKernelSetArgValue
  // CHECK-SAME: .hKernel = [[KERNEL_HANDLE1]]
  // CHECK-SAME: .argIndex = 2

  // CHECK: <--- urCommandBufferAppendKernelLaunchExp
  // CHECK-SAME: .hKernel = [[KERNEL_HANDLE1]]
  // CHECK-SAME: .numKernelAlternatives = 3
  // CHECK-SAME: .phKernelAlternatives = {[[KERNEL_HANDLE2:[0-9a-fA-Fx]+]], [[KERNEL_HANDLE3:[0-9a-fA-Fx]+]], [[KERNEL_HANDLE4:[0-9a-fA-Fx]+]]}
  auto DynamicCG =
      exp_ext::dynamic_command_group(Graph, {CGFA, CGFB, CGFC, CGFD});
  auto DynamicCGNode = Graph.add(DynamicCG);
  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  // Verify CGFA works with 3 arguments
  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.copy(Ptr, HostData.data(), Size).wait();
  int Ref = PatternA + PatternB;
  for (size_t i = 0; i < Size; i++) {
    assert(HostData[i] == Ref);
  }

  // Verify CGFB works with 2 arguments
  // CHECK: <--- urCommandBufferUpdateKernelLaunchExp
  // CHECK-SAME: .hNewKernel = [[KERNEL_HANDLE2]]
  // CHECK-SAME: .numNewMemObjArgs = 0
  // CHECK-SAME: .numNewPointerArgs = 1
  // CHECK-SAME: .numNewValueArgs = 1
  // CHECK-SAME: .stype = UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC
  // CHECK-SAME: .argIndex = 0
  // CHECK-SAME: .stype = UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC
  // CHECK-SAME: .argIndex = 1
  DynamicCG.set_active_cgf(1);
  ExecGraph.update(DynamicCGNode);
  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.copy(Ptr, HostData.data(), Size).wait();
  Ref = (PatternA + 1) + (PatternB + 1);
  for (size_t i = 0; i < Size; i++) {
    assert(HostData[i] == Ref);
  }

  // Verify CGFC works with 1 argument
  // CHECK: <--- urCommandBufferUpdateKernelLaunchExp
  // CHECK-SAME: .hNewKernel = [[KERNEL_HANDLE3]]
  // CHECK-SAME: .numNewMemObjArgs = 0
  // CHECK-SAME: .numNewPointerArgs = 1
  // CHECK-SAME: .numNewValueArgs = 0
  // CHECK-SAME: .stype = UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC
  // CHECK-SAME: .argIndex = 0
  DynamicCG.set_active_cgf(2);
  ExecGraph.update(DynamicCGNode);
  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.copy(Ptr, HostData.data(), Size).wait();
  Ref = PatternA - PatternB;
  for (size_t i = 0; i < Size; i++) {
    assert(HostData[i] == Ref);
  }

  // Verify CGFD works with 4 arguments
  // CHECK: <--- urCommandBufferUpdateKernelLaunchExp
  // CHECK-SAME: .hNewKernel = [[KERNEL_HANDLE4]]
  // CHECK-SAME: .numNewMemObjArgs = 0
  // CHECK-SAME: .numNewPointerArgs = 1
  // CHECK-SAME: .numNewValueArgs = 3
  // CHECK-SAME: .stype = UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_POINTER_ARG_DESC
  // CHECK-SAME: .argIndex = 0
  // CHECK-SAME: .stype = UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC
  // CHECK-SAME: .argIndex = 1
  // CHECK-SAME: .stype = UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC
  // CHECK-SAME: .argIndex = 2
  // CHECK-SAME: .stype = UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_UPDATE_VALUE_ARG_DESC
  // CHECK-SAME: .argIndex = 3
  DynamicCG.set_active_cgf(3);
  ExecGraph.update(DynamicCGNode);
  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.copy(Ptr, HostData.data(), Size).wait();
  Ref = PatternA + PatternB + PatternC;
  for (size_t i = 0; i < Size; i++) {
    assert(HostData[i] == Ref);
  }

  sycl::free(Ptr, Queue);

  return 0;
}
