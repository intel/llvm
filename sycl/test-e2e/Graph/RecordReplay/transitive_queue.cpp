// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Checks that the transitive queue recording feature is working as expected.
// i.e. submitting a command group function to a queue that has a dependency
// from a graph, should change the state of the queue to recording mode.

#include "../graph_common.hpp"

int main() {
  using T = int;

  device Dev;
  context Ctx{Dev};
  queue Q1{Ctx, Dev};
  queue Q2{Ctx, Dev};
  queue Q3{Ctx, Dev};

  const exp_ext::queue_state Recording = exp_ext::queue_state::recording;
  const exp_ext::queue_state Executing = exp_ext::queue_state::executing;

  auto assertQueueState = [&](exp_ext::queue_state ExpectedQ1,
                              exp_ext::queue_state ExpectedQ2,
                              exp_ext::queue_state ExpectedQ3) {
    assert(Q1.ext_oneapi_get_state() == ExpectedQ1);
    assert(Q2.ext_oneapi_get_state() == ExpectedQ2);
    assert(Q3.ext_oneapi_get_state() == ExpectedQ3);
  };

  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);

  T *PtrA = malloc_device<T>(Size, Q1);
  T *PtrB = malloc_device<T>(Size, Q1);
  T *PtrC = malloc_device<T>(Size, Q1);

  Q1.copy(DataA.data(), PtrA, Size);
  Q1.copy(DataB.data(), PtrB, Size);
  Q1.copy(DataC.data(), PtrC, Size);
  Q1.wait_and_throw();

  exp_ext::command_graph Graph{Q1.get_context(), Q1.get_device()};

  Graph.begin_recording(Q1);
  assertQueueState(Recording, Executing, Executing);

  auto GraphEventA = Q1.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> Id) { PtrA[Id]++; });
  });
  assertQueueState(Recording, Executing, Executing);

  // Since there is a dependency on GraphEventA which is part of a graph,
  // this will change Q2 to the recording state.
  auto GraphEventB = Q2.submit([&](handler &CGH) {
    CGH.depends_on(GraphEventA);
    CGH.parallel_for(range<1>(Size), [=](item<1> Id) { PtrA[Id]++; });
  });

  // Has no dependencies but should still be recorded to the graph because
  // the queue was implicitly changed to recording mode previously.
  auto GraphEventC = Q2.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> Id) { PtrB[Id]++; });
  });
  assertQueueState(Recording, Recording, Executing);

  // Q2 is now in recording mode. Submitting a command group to Q3 with a
  // dependency on an event from Q2 should change it to recording mode as well.
  auto GraphEventD = Q3.submit([&](handler &CGH) {
    CGH.depends_on(GraphEventB);
    CGH.parallel_for(range<1>(Size), [=](item<1> Id) { PtrC[Id]++; });
  });
  assertQueueState(Recording, Recording, Recording);

  Graph.end_recording(Q1);
  assertQueueState(Executing, Recording, Recording);
  Graph.end_recording(Q2);
  assertQueueState(Executing, Executing, Recording);

  auto GraphEventE = Q1.submit([&](handler &CGH) {
    CGH.depends_on(GraphEventD);
    CGH.parallel_for(range<1>(Size), [=](item<1> Id) { PtrC[Id]++; });
  });
  assertQueueState(Recording, Executing, Recording);

  Graph.end_recording(Q1);
  assertQueueState(Executing, Executing, Recording);

  // Q2 is not recording anymore. So this will be submitted outside the graph.
  auto OutsideEventA = Q2.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> Id) { PtrC[Id] /= 2; });
  });
  assertQueueState(Executing, Executing, Recording);

  try {
    // Q3 should still be recording. Adding a dependency from an event outside
    // the graph should fail.
    auto EventF = Q3.submit([&](handler &CGH) {
      CGH.depends_on(OutsideEventA);
      CGH.parallel_for(range<1>(Size), [=](item<1> Id) { PtrC[Id]++; });
    });
  } catch (exception &E) {
    assert(E.code() == sycl::errc::invalid);
    assertQueueState(Executing, Executing, Recording);
  }

  Q2.wait_and_throw();

  Q1.copy(PtrA, DataA.data(), Size);
  Q1.copy(PtrB, DataB.data(), Size);
  Q1.copy(PtrC, DataC.data(), Size);
  Q1.wait_and_throw();

  // Check that only DataC was changed before running the graph
  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceA[i], DataA[i], "DataA"));
    assert(check_value(i, ReferenceB[i], DataB[i], "DataB"));
    assert(check_value(i, ReferenceC[i] / 2, DataC[i], "DataC"));
  }

  Graph.end_recording();
  assertQueueState(Executing, Executing, Executing);

  auto GraphExec = Graph.finalize();

  Q1.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  Q1.wait_and_throw();

  Q1.copy(PtrA, DataA.data(), Size);
  Q1.copy(PtrB, DataB.data(), Size);
  Q1.copy(PtrC, DataC.data(), Size);
  Q1.wait_and_throw();

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceA[i] + 2, DataA[i], "DataA"));
    assert(check_value(i, ReferenceB[i] + 1, DataB[i], "DataB"));
    assert(check_value(i, ReferenceC[i] / 2 + 2, DataC[i], "DataC"));
  }

  return 0;
}
