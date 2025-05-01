// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// REQUIRES: aspect-usm_shared_allocations

// Tests using dynamic command groups to update a host task node
#include "../graph_common.hpp"

using T = int;

void calculate_reference_data(size_t Iterations, size_t Size,
                              std::vector<T> &ReferenceA,
                              std::vector<T> &ReferenceB,
                              std::vector<T> &ReferenceC, T ModValue) {
  for (size_t n = 0; n < Iterations; n++) {
    for (size_t i = 0; i < Size; i++) {
      ReferenceA[i]++;
      ReferenceB[i] += ReferenceA[i];
      ReferenceC[i] -= ReferenceA[i];
      ReferenceB[i]--;
      ReferenceC[i]--;
      ReferenceC[i] += ModValue;
      ReferenceC[i] *= 2;
    }
  }
}

int main() {
  queue Queue{};

  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  auto DataA2 = DataA;
  auto DataB2 = DataB;
  auto DataC2 = DataC;

  const T ModValue = 7;
  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  calculate_reference_data(Iterations, Size, ReferenceA, ReferenceB, ReferenceC,
                           ModValue);
  // We're not updating the buffers here so we need to stack the reference
  // calculations for after the host-task update
  std::vector<T> ReferenceA2(ReferenceA), ReferenceB2(ReferenceB),
      ReferenceC2(ReferenceC);
  calculate_reference_data(Iterations, Size, ReferenceA2, ReferenceB2,
                           ReferenceC2, ModValue * 2);

  T *PtrA = sycl::malloc_device<T>(Size, Queue);
  T *PtrB = sycl::malloc_device<T>(Size, Queue);
  T *PtrC = sycl::malloc_shared<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.copy(DataC.data(), PtrC, Size);
  Queue.wait_and_throw();

  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

  // Add some commands to the graph
  auto LastOperation = add_kernels_usm(Graph, Size, PtrA, PtrB, PtrC);

  // Create two different command groups for the host task
  auto CGFA = [&](handler &CGH) {
    CGH.host_task([=]() {
      for (size_t i = 0; i < Size; i++) {
        PtrC[i] += ModValue;
      }
    });
  };
  auto CGFB = [&](handler &CGH) {
    CGH.host_task([=]() {
      for (size_t i = 0; i < Size; i++) {
        PtrC[i] += ModValue * 2;
      }
    });
  };

  auto DynamicCG = exp_ext::dynamic_command_group(Graph, {CGFA, CGFB});
  // Add a host task which modifies PtrC using the dynamic CG
  auto HostTaskNode =
      Graph.add(DynamicCG, exp_ext::property::node::depends_on{LastOperation});

  // Add another node that depends on the host-task
  Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>(Size), [=](item<1> Item) {
          PtrC[Item.get_linear_id()] *= 2;
        });
      },
      exp_ext::property::node::depends_on{HostTaskNode});

  auto GraphExec = Graph.finalize(exp_ext::property::graph::updatable{});

  // Execute several Iterations of the graph with the first host-task (CGFA)
  event Event;
  for (unsigned n = 0; n < Iterations; n++) {
    Event = Queue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExec);
    });
  }
  Queue.wait_and_throw();
  Queue.copy(PtrA, DataA.data(), Size);
  Queue.copy(PtrB, DataB.data(), Size);
  Queue.copy(PtrC, DataC.data(), Size);
  Queue.wait_and_throw();
  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceA[i], DataA[i], "DataA"));
    assert(check_value(i, ReferenceB[i], DataB[i], "DataB"));
    assert(check_value(i, ReferenceC[i], DataC[i], "DataC"));
  }

  // Update to CGFB
  DynamicCG.set_active_index(1);
  GraphExec.update(HostTaskNode);

  // Execute several Iterations of the graph for second host-task (CGFB)
  for (unsigned n = 0; n < Iterations; n++) {
    Event = Queue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExec);
    });
  }

  Queue.wait_and_throw();
  Queue.copy(PtrA, DataA2.data(), Size);
  Queue.copy(PtrB, DataB2.data(), Size);
  Queue.copy(PtrC, DataC2.data(), Size);
  Queue.wait_and_throw();
  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceA2[i], DataA2[i], "DataA2"));
    assert(check_value(i, ReferenceB2[i], DataB2[i], "DataB2"));
    assert(check_value(i, ReferenceC2[i], DataC2[i], "DataC2"));
  }

  sycl::free(PtrA, Queue);
  sycl::free(PtrB, Queue);
  sycl::free(PtrC, Queue);
  return 0;
}
