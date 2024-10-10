// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Regression test for using transitive queue recording when a graph
// event is passed as a dependency to a barrier operation in a different
// in-order queue.

#include "../graph_common.hpp"
#include <sycl/properties/all_properties.hpp>

int main() {
  using T = int;

  device Dev;
  context Ctx{Dev};

  property_list InOrderProp = {property::queue::in_order{}};
  queue Q1{Ctx, Dev, InOrderProp};
  queue Q2{Ctx, Dev, InOrderProp};

  const exp_ext::queue_state Recording = exp_ext::queue_state::recording;
  const exp_ext::queue_state Executing = exp_ext::queue_state::executing;

  auto assertQueueState = [&](exp_ext::queue_state ExpectedQ1,
                              exp_ext::queue_state ExpectedQ2) {
    assert(Q1.ext_oneapi_get_state() == ExpectedQ1);
    assert(Q2.ext_oneapi_get_state() == ExpectedQ2);
  };

  T *PtrA = malloc_device<T>(Size, Q1);
  T *PtrB = malloc_device<T>(Size, Q1);
  T *PtrC = malloc_device<T>(Size, Q1);

  exp_ext::command_graph Graph{Q1.get_context(), Q1.get_device()};

  Graph.begin_recording(Q1);
  assertQueueState(Recording, Executing);

  T PatternA = 42;
  auto EventA =
      Q1.submit([&](handler &CGH) { CGH.fill(PtrA, PatternA, Size); });
  assertQueueState(Recording, Executing);

  T PatternB = 0xA;
  auto EventB = Q1.fill(PtrB, PatternB, Size);
  assertQueueState(Recording, Executing);

  auto Barrier1 = Q1.ext_oneapi_submit_barrier();
  assertQueueState(Recording, Executing);

  // Depends on Q1 barrier, should put Q2 in recording state
  auto Barrier = Q2.ext_oneapi_submit_barrier({Barrier1});
  assertQueueState(Recording, Recording);

  // Q2 is now in recording state
  auto EventC = Q2.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size),
                     [=](item<1> Id) { PtrC[Id] = PtrA[Id] + PtrB[Id]; });
  });
  assertQueueState(Recording, Recording);

  Graph.end_recording();
  assertQueueState(Executing, Executing);

  auto GraphExec = Graph.finalize();

  Q1.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  Q1.wait_and_throw();

  std::vector<T> DataA(Size), DataB(Size), DataC(Size);
  Q1.copy(PtrA, DataA.data(), Size);
  Q1.copy(PtrB, DataB.data(), Size);
  Q1.copy(PtrC, DataC.data(), Size);
  Q1.wait_and_throw();

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, PatternA, DataA[i], "DataA"));
    assert(check_value(i, PatternB, DataB[i], "DataB"));
    assert(check_value(i, (PatternA + PatternB), DataC[i], "DataC"));
  }

  return 0;
}
