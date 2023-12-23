// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

#include "../graph_common.hpp"

//// Test Explicit API graph construction with USM.
///
/// @param Q Command-queue to make kernel submissions to.
/// @param Size Number of elements in the buffers.
/// @param DataA Pointer to first USM allocation to use in kernels.
/// @param DataB Pointer to second USM allocation to use in kernels.
/// @param DataC Pointer to third USM allocation to use in kernels.
///
/// @return Event corresponding to the exit node of the submission sequence.
template <typename T>
event run_kernels_usm_with_barrier(queue Q, const size_t Size, T *DataA,
                                   T *DataB, T *DataC) {
  // Read & write Buffer A
  auto EventA = Q.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> Id) {
      auto LinID = Id.get_linear_id();
      DataA[LinID]++;
    });
  });

  Q.ext_oneapi_submit_barrier();

  // Reads Buffer A
  // Read & Write Buffer B
  auto EventB = Q.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> Id) {
      auto LinID = Id.get_linear_id();
      DataB[LinID] += DataA[LinID];
    });
  });

  // Reads Buffer A
  // Read & writes Buffer C
  auto EventC = Q.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> Id) {
      auto LinID = Id.get_linear_id();
      DataC[LinID] -= DataA[LinID];
    });
  });

  Q.ext_oneapi_submit_barrier();

  // Read & write Buffers B and C
  auto ExitEvent = Q.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> Id) {
      auto LinID = Id.get_linear_id();
      DataB[LinID]--;
      DataC[LinID]--;
    });
  });
  return ExitEvent;
}

int main() {
  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  using T = int;

  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  calculate_reference_data(Iterations, Size, ReferenceA, ReferenceB,
                           ReferenceC);

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);
  T *PtrC = malloc_device<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.copy(DataC.data(), PtrC, Size);
  Queue.wait_and_throw();

  // Add commands to graph
  Graph.begin_recording(Queue);
  auto ev = run_kernels_usm_with_barrier(Queue, Size, PtrA, PtrB, PtrC);
  Graph.end_recording(Queue);

  auto GraphExec = Graph.finalize();

  event Event;
  for (unsigned n = 0; n < Iterations; n++) {
    Event =
        Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  }
  Queue.wait_and_throw();

  Queue.copy(PtrA, DataA.data(), Size);
  Queue.copy(PtrB, DataB.data(), Size);
  Queue.copy(PtrC, DataC.data(), Size);
  Queue.wait_and_throw();

  free(PtrA, Queue);
  free(PtrB, Queue);
  free(PtrC, Queue);

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceA[i], DataA[i], "DataA"));
    assert(check_value(i, ReferenceB[i], DataB[i], "DataB"));
    assert(check_value(i, ReferenceC[i], DataC[i], "DataC"));
  }

  return 0;
}
