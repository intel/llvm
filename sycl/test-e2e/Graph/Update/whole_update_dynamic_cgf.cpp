// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests interaction of whole graph update and dynamic command-groups

#include "../graph_common.hpp"

int main() {
  queue Queue{};
  exp_ext::command_graph GraphA{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph GraphB{Queue.get_context(), Queue.get_device()};

  int *Ptr = malloc_device<int>(Size, Queue);
  std::vector<int> HostData(Size);

  int PatternA = 42;
  auto CGFA = [&](handler &CGH) {
    CGH.parallel_for(Size,
                     [=](item<1> Item) { Ptr[Item.get_id()] = PatternA; });
  };

  int PatternB = 0xA;
  auto CGFB = [&](handler &CGH) {
    CGH.parallel_for(Size,
                     [=](item<1> Item) { Ptr[Item.get_id()] = PatternB; });
  };

  auto DynamicCGA = exp_ext::dynamic_command_group(GraphA, {CGFA, CGFB});
  auto DynamicCGNodeA = GraphA.add(DynamicCGA);

  auto DynamicCGB = exp_ext::dynamic_command_group(GraphB, {CGFA, CGFB});
  auto DynamicCGNodeB = GraphB.add(DynamicCGB);
  DynamicCGB.set_active_cgf(1); //  Check if doesn't affect GraphA

  auto ExecGraph = GraphA.finalize(exp_ext::property::graph::updatable{});

  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.copy(Ptr, HostData.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostData[i] == PatternA);
  }

  // Graph B has CGF B as active, while Graph A has CGF A as active.
  // Different command-groups should error due to being different
  // kernels.
  std::error_code ErrorCode = make_error_code(sycl::errc::success);
  try {
    ExecGraph.update(GraphB);
  } catch (const sycl::exception &e) {
    ErrorCode = e.code();
  }
  assert(ErrorCode == sycl::errc::invalid);

  // Both ExecGraph and Graph B have CGFB as active, so
  // whole graph update should be valid as graphs match.
  DynamicCGA.set_active_cgf(1);
  ExecGraph.update(DynamicCGNodeA);
  ExecGraph.update(GraphB);

  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.copy(Ptr, HostData.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostData[i] == PatternB);
  }

  sycl::free(Ptr, Queue);

  return 0;
}
