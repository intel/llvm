// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests that graph update works for kernel nodes whose parameters are large
// structs with padding (added by the compiler).
#include "../graph_common.hpp"

int main() {

  constexpr size_t Size = 8;

  /* Since val1 is smaller than val2, the compiler should automatically add
   * padding to val1 */
  struct PaddedStruct {
    uint8_t val1 = 20;
    size_t val2[Size] = {1, 2, 3, 4, 5, 6, 7, 8};
  } PaddedKernelParam;

  static_assert(sizeof(PaddedStruct) == sizeof(size_t) * (Size + 1));

  queue Queue{};
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  int *PtrA = malloc_device<int>(Size, Queue);
  Queue.memset(PtrA, 0, Size * sizeof(int)).wait();

  auto KernelNode = Graph.add([&](handler &cgh) {
    cgh.parallel_for(Size, [=](item<1> Item) {
      size_t GlobalID = Item.get_id();
      PtrA[GlobalID] +=
          PaddedKernelParam.val1 * PaddedKernelParam.val2[GlobalID];
    });
  });

  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});
  Queue.ext_oneapi_graph(ExecGraph).wait();

  std::vector<int> HostDataA(Size);
  Queue.copy(PtrA, HostDataA.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostDataA[i] == PaddedKernelParam.val1 * PaddedKernelParam.val2[i]);
  }

  return 0;
}
