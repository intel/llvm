// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests the enqueue free function kernel shortcuts.

#include "../graph_common.hpp"
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>

int main() {
  queue InOrderQueue{property::queue::in_order{}};

  using T = int;

  T *PtrA = malloc_device<T>(Size, InOrderQueue);
  T *PtrB = malloc_device<T>(Size, InOrderQueue);
  T *PtrC = malloc_device<T>(Size, InOrderQueue);

  exp_ext::command_graph Graph{InOrderQueue};
  Graph.begin_recording(InOrderQueue);

  T Pattern = 42;
  exp_ext::fill(InOrderQueue, PtrA, Pattern, Size);

  exp_ext::single_task(InOrderQueue, [=]() {
    for (size_t i = 0; i < Size; ++i) {
      PtrB[i] = i;
    }
  });

  exp_ext::parallel_for(
      InOrderQueue, sycl::range<1>{Size},
      [=](sycl::item<1> Item) { PtrC[Item] = PtrA[Item] * PtrB[Item]; });

  std::vector<T> Output(Size);
  exp_ext::copy(InOrderQueue, PtrC, Output.data(), Size);

  Graph.end_recording();

  auto GraphExec = Graph.finalize();

  exp_ext::execute_graph(InOrderQueue, GraphExec);
  InOrderQueue.wait_and_throw();

  free(PtrA, InOrderQueue);
  free(PtrB, InOrderQueue);
  free(PtrC, InOrderQueue);

  for (size_t i = 0; i < Size; i++) {
    T Ref = Pattern * i;
    assert(Output[i] == Ref);
  }

  return 0;
}
