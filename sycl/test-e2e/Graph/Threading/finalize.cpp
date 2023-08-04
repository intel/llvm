// REQUIRES: level_zero, gpu
// RUN: %{build_pthread_inc} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Test finalizing and submitting a graph in a threaded situation.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"
#include <detail/graph_impl.hpp>

#include <thread>

bool checkExecGraphSchedule(
    std::shared_ptr<sycl::ext::oneapi::experimental::detail::exec_graph_impl>
        GraphA,
    std::shared_ptr<sycl::ext::oneapi::experimental::detail::exec_graph_impl>
        GraphB) {
  auto ScheduleA = GraphA->getSchedule();
  auto ScheduleB = GraphB->getSchedule();
  if (ScheduleA.size() != ScheduleB.size())
    return false;

  std::vector<
      std::shared_ptr<sycl::ext::oneapi::experimental::detail::node_impl>>
      VScheduleA{std::begin(ScheduleA), std::end(ScheduleA)};
  std::vector<
      std::shared_ptr<sycl::ext::oneapi::experimental::detail::node_impl>>
      VScheduleB{std::begin(ScheduleB), std::end(ScheduleB)};

  for (size_t i = 0; i < VScheduleA.size(); i++) {
    if (!VScheduleA[i]->isSimilar(VScheduleB[i]))
      return false;
  }
  return true;
}

int main() {
  queue Queue;

  using T = int;

  const unsigned NumThreads = std::thread::hardware_concurrency();
  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  calculate_reference_data(NumThreads, Size, ReferenceA, ReferenceB,
                           ReferenceC);

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);
  T *PtrB = malloc_device<T>(Size, Queue);
  T *PtrC = malloc_device<T>(Size, Queue);

  Queue.copy(DataA.data(), PtrA, Size);
  Queue.copy(DataB.data(), PtrB, Size);
  Queue.copy(DataC.data(), PtrC, Size);
  Queue.wait_and_throw();

  Graph.begin_recording(Queue);
  run_kernels_usm(Queue, Size, PtrA, PtrB, PtrC);
  Graph.end_recording();

  Barrier SyncPoint{NumThreads};

  std::map<int, exp_ext::command_graph<exp_ext::graph_state::executable>>
      GraphsExecMap;
  auto FinalizeGraph = [&](int ThreadNum) {
    SyncPoint.wait();
    auto GraphExec = Graph.finalize();
    GraphsExecMap.insert(
        std::map<int,
                 exp_ext::command_graph<exp_ext::graph_state::executable>>::
            value_type(ThreadNum, GraphExec));
    Queue.submit([&](sycl::handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  };

  std::vector<std::thread> Threads;
  Threads.reserve(NumThreads);

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads.emplace_back(FinalizeGraph, i);
  }

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads[i].join();
  }

  Queue.wait_and_throw();

  Queue.copy(PtrA, DataA.data(), Size);
  Queue.copy(PtrB, DataB.data(), Size);
  Queue.copy(PtrC, DataC.data(), Size);
  Queue.wait_and_throw();

  // Ref computation
  queue QueueRef{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph GraphRef{Queue.get_context(), Queue.get_device()};

  T *PtrARef = malloc_device<T>(Size, QueueRef);
  T *PtrBRef = malloc_device<T>(Size, QueueRef);
  T *PtrCRef = malloc_device<T>(Size, QueueRef);

  QueueRef.copy(DataA.data(), PtrARef, Size);
  QueueRef.copy(DataB.data(), PtrBRef, Size);
  QueueRef.copy(DataC.data(), PtrCRef, Size);
  QueueRef.wait_and_throw();

  GraphRef.begin_recording(QueueRef);
  run_kernels_usm(QueueRef, Size, PtrA, PtrB, PtrC);
  GraphRef.end_recording();

  for (unsigned i = 0; i < NumThreads; ++i) {
    auto GraphExecRef = GraphRef.finalize();
    QueueRef.submit(
        [&](sycl::handler &CGH) { CGH.ext_oneapi_graph(GraphExecRef); });
    auto GraphExecImpl =
        sycl::detail::getSyclObjImpl(GraphsExecMap.find(i)->second);
    auto GraphExecRefImpl = sycl::detail::getSyclObjImpl(GraphExecRef);
    assert(checkExecGraphSchedule(GraphExecImpl, GraphExecRefImpl));
  }

  free(PtrARef, QueueRef);
  free(PtrBRef, QueueRef);
  free(PtrCRef, QueueRef);

  free(PtrA, Queue);
  free(PtrB, Queue);
  free(PtrC, Queue);

  assert(ReferenceA == DataA);
  assert(ReferenceB == DataB);
  assert(ReferenceC == DataC);

  return 0;
}
