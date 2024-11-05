// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests the enqueue free function using USM and submit_with_event for
// dependencies

#include "../graph_common.hpp"
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

int main() {
  queue Queue{};

  using T = int;

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);
  T *PtrC = malloc_device<T>(Size, Queue);

  exp_ext::command_graph Graph{Queue};
  Graph.begin_recording(Queue);

  T Pattern = 42;
  event NodeA = exp_ext::submit_with_event(
      Queue, [&](handler &CGH) { exp_ext::fill(CGH, PtrA, Pattern, Size); });

  event NodeB = exp_ext::submit_with_event(Queue, [&](handler &CGH) {
    exp_ext::single_task(CGH, [=]() {
      for (size_t i = 0; i < Size; ++i) {
        PtrB[i] = i;
      }
    });
  });

  event NodeC = exp_ext::submit_with_event(Queue, [&](handler &CGH) {
    CGH.depends_on({NodeA, NodeB});
    exp_ext::parallel_for(CGH, range<1>{Size}, [=](item<1> Item) {
      PtrC[Item] = PtrA[Item] * PtrB[Item];
    });
  });

  std::vector<T> Output(Size);
  exp_ext::submit_with_event(Queue, [&](handler &CGH) {
    CGH.depends_on(NodeC);
    exp_ext::copy(CGH, PtrC, Output.data(), Size);
  });

  Graph.end_recording();

  auto GraphExec = Graph.finalize();

  exp_ext::submit_with_event(Queue, [&](handler &CGH) {
    exp_ext::execute_graph(CGH, GraphExec);
  }).wait();

  free(PtrA, Queue);
  free(PtrB, Queue);
  free(PtrC, Queue);

  for (size_t i = 0; i < Size; i++) {
    T Ref = Pattern * i;
    assert(Output[i] == Ref);
  }

  return 0;
}
