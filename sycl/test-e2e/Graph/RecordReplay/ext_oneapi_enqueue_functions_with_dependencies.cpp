// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests the enqueue free function kernel shortcuts.

#include "../graph_common.hpp"
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>

int main() {
  device Device{};
  context Context{Device};

  queue InOrderQueue{Context, Device, property::queue::in_order{}};
  queue OtherQueue{Context, Device, property::queue::in_order{}};

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
      [=](sycl::item<1> Item) { PtrC[Item] += PtrA[Item] * PtrB[Item]; });

  std::vector<T> Output(Size);
  exp_ext::copy(InOrderQueue, PtrC, Output.data(), Size);

  Graph.end_recording();

  auto GraphExec = Graph.finalize();

  const size_t MemsetValue = 12;
  sycl::event Event =
      exp_ext::submit_with_event(OtherQueue, [&](sycl::handler &CGH) {
        exp_ext::single_task(CGH, [=]() {
          for (size_t I = 0; I < Size; ++I)
            PtrC[I] = MemsetValue;
        });
      });

  exp_ext::submit(InOrderQueue, [&](sycl::handler &CGH) {
    CGH.depends_on(Event);
    exp_ext::execute_graph(CGH, GraphExec);
  });

  InOrderQueue.wait_and_throw();

  free(PtrA, InOrderQueue);
  free(PtrB, InOrderQueue);
  free(PtrC, InOrderQueue);

  for (size_t i = 0; i < Size; i++) {
    T Ref = Pattern * i + MemsetValue;
    assert(check_value(i, Ref, Output[i], "Output"));
  }

  return 0;
}
