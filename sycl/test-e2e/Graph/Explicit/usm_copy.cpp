// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected fail as memcopy not implemented yet
// XFAIL: *

// Tests adding a usm memcpy node using the explicit API and submitting
// the graph.

#include "../graph_common.hpp"

int main() {
  queue TestQueue;

  using T = int;

  const T ModValue = 7;
  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  // Create reference data for output
  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  for (unsigned i = 0; i < Iterations; i++) {
    for (size_t j = 0; j < Size; j++) {
      ReferenceA[j] = ReferenceB[j];
      ReferenceA[j] += ModValue;
      ReferenceB[j] = ReferenceA[j];
      ReferenceB[j] += ModValue;
      ReferenceC[j] = ReferenceB[j];
    }
  }

  exp_ext::command_graph Graph{TestQueue.get_context(), TestQueue.get_device()};

  T *PtrA = malloc_device<T>(Size, TestQueue);
  T *PtrB = malloc_device<T>(Size, TestQueue);
  T *PtrC = malloc_device<T>(Size, TestQueue);

  TestQueue.copy(DataA.data(), PtrA, Size);
  TestQueue.copy(DataB.data(), PtrB, Size);
  TestQueue.copy(DataC.data(), PtrC, Size);
  TestQueue.wait_and_throw();

  // memcpy from B to A
  auto NodeA = Graph.add([&](handler &CGH) { CGH.copy(PtrB, PtrA, Size); });

  // Read & write A
  auto NodeB = Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>(Size), [=](item<1> id) {
          auto LinID = id.get_linear_id();
          PtrA[LinID] += ModValue;
        });
      },
      {exp_ext::property::node::depends_on(NodeA)});

  // memcpy from B to A
  auto NodeC = Graph.add([&](handler &CGH) { CGH.copy(PtrA, PtrB, Size); },
                         {exp_ext::property::node::depends_on(NodeB)});

  // Read and write B
  auto nodeD = Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>(Size), [=](item<1> id) {
          auto LinID = id.get_linear_id();
          PtrB[LinID] += ModValue;
        });
      },
      {exp_ext::property::node::depends_on(NodeC)});

  // memcpy from B to C
  Graph.add([&](handler &CGH) { CGH.copy(PtrB, PtrC, Size); },
            {exp_ext::property::node::depends_on(NodeB)});

  auto GraphExec = Graph.finalize();

  event Event;
  for (unsigned n = 0; n < Iterations; n++) {
    Event = TestQueue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExec);
    });
  }

  TestQueue.wait_and_throw();

  TestQueue.copy(PtrA, DataA.data(), Size);
  TestQueue.copy(PtrB, DataB.data(), Size);
  TestQueue.copy(PtrC, DataC.data(), Size);

  free(PtrA, TestQueue);
  free(PtrB, TestQueue);
  free(PtrC, TestQueue);

  assert(ReferenceA == DataA);
  assert(ReferenceB == DataB);
  assert(ReferenceC == DataC);

  return 0;
}
