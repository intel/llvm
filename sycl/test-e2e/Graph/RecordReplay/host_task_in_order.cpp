// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero && linux %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//

// This test uses a host_task when adding a command_graph node to an
// in-order queue.

#include "../graph_common.hpp"

int main() {
  queue Queue{{property::queue::in_order{},
               sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  using T = int;

  if (!Queue.get_device().has(sycl::aspect::usm_shared_allocations)) {
    return 0;
  }

  const T ModValue = T{7};
  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  std::vector<T> Reference(DataC);
  for (unsigned n = 0; n < Iterations; n++) {
    for (size_t i = 0; i < Size; i++) {
      Reference[i] = (((DataA[i] + DataB[i]) * ModValue) + 1) * DataB[i];
    }
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);
  T *PtrC = malloc_shared<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.copy(DataC.data(), PtrC, Size);
  Queue.wait_and_throw();

  Graph.begin_recording(Queue);

  // Vector add to output
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> id) { PtrC[id] = PtrA[id]; });
  });

  // Vector add to output
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> id) { PtrC[id] += PtrB[id]; });
  });

  // Modify the output values in a host_task
  Queue.submit([&](handler &CGH) {
    CGH.host_task([=]() {
      for (size_t i = 0; i < Size; i++) {
        PtrC[i] *= ModValue;
      }
    });
  });

  // Modify temp buffer and write to output buffer
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> id) { PtrC[id] += 1; });
  });

  // Modify temp buffer and write to output buffer
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> id) { PtrC[id] *= PtrB[id]; });
  });

  Graph.end_recording(Queue);

  auto GraphExec = Graph.finalize();

  event Event;
  for (unsigned n = 0; n < Iterations; n++) {
    Event = Queue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExec);
    });
    Event.wait();
  }
  Queue.wait_and_throw();

  Queue.copy(PtrC, DataC.data(), Size);
  Queue.wait_and_throw();

  free(PtrA, Queue);
  free(PtrB, Queue);
  free(PtrC, Queue);

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, Reference[i], DataC[i], "DataC"));
  }

  return 0;
}
