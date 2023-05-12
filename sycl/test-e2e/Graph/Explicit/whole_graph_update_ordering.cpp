// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected fail as executable graph update and host tasks both aren't
// implemented.
// XFAIL: *

// Tests executable graph update by introducing a delay in to the update
// transactions dependencies to check correctness of behaviour.

#include "../graph_common.hpp"
#include <thread>

int main() {
  queue TestQueue;

  using T = int;

  if (!TestQueue.get_device().has(sycl::aspect::usm_shared_allocations)) {
    return 0;
  }

  std::vector<T> DataA(Size), DataB(Size), DataC(Size);
  std::vector<T> HostTaskOutput(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  auto DataA2 = DataA;
  auto DataB2 = DataB;
  auto DataC2 = DataC;

  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  calculate_reference_data(Iterations, Size, ReferenceA, ReferenceB,
                           ReferenceC);

  exp_ext::command_graph GraphA{TestQueue.get_context(),
                                TestQueue.get_device()};

  T *PtrA = malloc_shared<T>(Size, TestQueue);
  T *PtrB = malloc_shared<T>(Size, TestQueue);
  T *PtrC = malloc_shared<T>(Size, TestQueue);
  T *PtrOut = malloc_shared<T>(Size, TestQueue);

  TestQueue.copy(DataA.data(), PtrA, Size);
  TestQueue.copy(DataB.data(), PtrB, Size);
  TestQueue.copy(DataC.data(), PtrC, Size);
  TestQueue.wait_and_throw();

  // Add commands to first graph
  auto NodeA = add_kernels_usm(GraphA, Size, PtrA, PtrB, PtrC);

  // host task to induce a wait for dependencies
  GraphA.add(
      [&](handler &CGH) {
        CGH.host_task([=]() {
          for (size_t i = 0; i < Size; i++) {
            PtrOut[i] = PtrC[i];
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(500));
        });
      },
      {exp_ext::property::node::depends_on(NodeA)});

  auto GraphExec = GraphA.finalize();

  exp_ext::command_graph GraphB{TestQueue.get_context(),
                                TestQueue.get_device()};

  T *PtrA2 = malloc_shared<T>(Size, TestQueue);
  T *PtrB2 = malloc_shared<T>(Size, TestQueue);
  T *PtrC2 = malloc_shared<T>(Size, TestQueue);

  TestQueue.copy(DataA2.data(), PtrA2, Size);
  TestQueue.copy(DataB2.data(), PtrB2, Size);
  TestQueue.copy(DataC2.data(), PtrC2, Size);
  TestQueue.wait_and_throw();

  // Adds commands to second graph
  auto NodeB = add_kernels_usm(GraphB, Size, PtrA2, PtrB2, PtrC2);

  // host task to match the graph topology, but we don't need to sleep this
  // time because there is no following update.
  GraphB.add(
      [&](handler &CGH) {
        // This should be access::target::host_task but it has not been
        // implemented yet.
        CGH.host_task([=]() {
          for (size_t i = 0; i < Size; i++) {
            PtrOut[i] = PtrC2[i];
          }
        });
      },
      {exp_ext::property::node::depends_on(NodeB)});

  event Event;
  for (unsigned n = 0; n < Iterations; n++) {
    Event = TestQueue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExec);
    });
  }

  GraphExec.update(GraphB);

  // Execute several Iterations of the graph for 2nd set of buffers
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
  TestQueue.copy(PtrOut, HostTaskOutput.data(), Size);

  TestQueue.copy(PtrA2, DataA.data(), Size);
  TestQueue.copy(PtrB2, DataB.data(), Size);
  TestQueue.copy(PtrC2, DataC.data(), Size);
  TestQueue.wait_and_throw();

  free(PtrA, TestQueue);
  free(PtrB, TestQueue);
  free(PtrC, TestQueue);
  free(PtrOut, TestQueue);

  free(PtrA2, TestQueue);
  free(PtrB2, TestQueue);
  free(PtrC2, TestQueue);

  assert(ReferenceA == DataA);
  assert(ReferenceB == DataB);
  assert(ReferenceC == DataC);
  assert(ReferenceC == HostTaskOutput);

  assert(ReferenceA == DataA2);
  assert(ReferenceB == DataB2);
  assert(ReferenceC == DataC2);

  return 0;
}
