// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//

// Tests that the optimization to use the L0 Copy Engine for memory commands
// does not interfere with the linear graph optimization

#include "../graph_common.hpp"

#include <sycl/properties/queue_properties.hpp>

int main() {
  queue Queue{{sycl::property::queue::in_order{}}};

  using T = int;

  const T ModValue = 7;
  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  // Create reference data for output
  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  for (size_t i = 0; i < Iterations; i++) {
    for (size_t j = 0; j < Size; j++) {
      ReferenceA[j] += ModValue;
      ReferenceB[j] = ReferenceA[j];
      ReferenceB[j] -= ModValue;
      ReferenceC[j] = ReferenceB[j];
      ReferenceC[j] += ModValue;
    }
  }

  ext::oneapi::experimental::command_graph Graph{Queue.get_context(),
                                                 Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);
  T *PtrC = malloc_device<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.copy(DataC.data(), PtrC, Size);
  Queue.wait_and_throw();

  Graph.begin_recording(Queue);
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> id) {
      auto LinID = id.get_linear_id();
      PtrA[LinID] += ModValue;
    });
  });

  Queue.submit([&](handler &CGH) { CGH.memcpy(PtrB, PtrA, Size * sizeof(T)); });

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> id) {
      auto LinID = id.get_linear_id();
      PtrB[LinID] -= ModValue;
    });
  });

  Queue.submit([&](handler &CGH) { CGH.memcpy(PtrC, PtrB, Size * sizeof(T)); });

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> id) {
      auto LinID = id.get_linear_id();
      PtrC[LinID] += ModValue;
    });
  });

  Graph.end_recording();

  auto GraphExec = Graph.finalize();

  event Event;
  for (unsigned n = 0; n < Iterations; n++) {
    Event =
        Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  }

  Queue.copy(PtrA, DataA.data(), Size, Event);
  Queue.copy(PtrB, DataB.data(), Size, Event);
  Queue.copy(PtrC, DataC.data(), Size, Event);
  Queue.wait_and_throw();

  free(PtrA, Queue);
  free(PtrB, Queue);
  free(PtrC, Queue);

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceA[i], DataA[i], "DataA"));
    assert(check_value(i, ReferenceB[i], DataB[i], "DataB"));
    assert(check_value(i, ReferenceC[i], DataC[i], "DataC"));
  }
}
