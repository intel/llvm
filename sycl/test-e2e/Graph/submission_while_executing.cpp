// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}

// https://github.com/intel/llvm/issues/11277:
// UNSUPPORTED: windows
//
// CHECK-NOT: LEAK

// Test calling queue::submit(graph) while the previous submission of graph has
// not been completed. The second run is to check that there are no leaks
// reported with the embedded ZE_DEBUG=4 testing capability.

#include "graph_common.hpp"

int main() {
  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  using T = int;

  size_t LargeSize =
      1000000; // we use large Size to increase the kernel execution time
  size_t NumIterations = 10;
  size_t SuccessfulSubmissions = 0;

  std::vector<T> DataA(LargeSize), DataB(LargeSize), DataC(LargeSize);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(LargeSize, Queue);
  T *PtrB = malloc_device<T>(LargeSize, Queue);
  T *PtrC = malloc_device<T>(LargeSize, Queue);

  Queue.copy(DataA.data(), PtrA, LargeSize);
  Queue.copy(DataB.data(), PtrB, LargeSize);
  Queue.copy(DataC.data(), PtrC, LargeSize);
  Queue.wait_and_throw();

  Graph.begin_recording(Queue);
  run_kernels_usm(Queue, LargeSize, PtrA, PtrB, PtrC);
  Graph.end_recording();

  auto GraphExec = Graph.finalize();

  // Serial Submissions
  for (unsigned i = 0; i < NumIterations; ++i) {
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
    Queue.wait_and_throw();
  }

  // Concurrent Submissions
  sycl::event Event;
  sycl::info::event_command_status PreEventInfo =
      sycl::info::event_command_status::ext_oneapi_unknown;
  std::error_code ErrorCode = make_error_code(sycl::errc::success);
  for (unsigned i = 0; i < NumIterations; ++i) {
    try {
      Event =
          Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
    } catch (const sycl::exception &e) {
      ErrorCode = e.code();
    }
    if ((PreEventInfo == sycl::info::event_command_status::submitted) ||
        (PreEventInfo == sycl::info::event_command_status::running)) {
      assert(ErrorCode == sycl::errc::invalid);
    } else {
      // Submission has succeeded
      SuccessfulSubmissions++;
    }
    PreEventInfo =
        Event.get_info<sycl::info::event::command_execution_status>();
  }
  Queue.wait_and_throw();

  Queue.copy(PtrA, DataA.data(), LargeSize);
  Queue.copy(PtrB, DataB.data(), LargeSize);
  Queue.copy(PtrC, DataC.data(), LargeSize);
  Queue.wait_and_throw();

  free(PtrA, Queue);
  free(PtrB, Queue);
  free(PtrC, Queue);

  // Compute the reference based on the total number of successful executions
  calculate_reference_data(NumIterations + SuccessfulSubmissions, LargeSize,
                           ReferenceA, ReferenceB, ReferenceC);
  assert(ReferenceA == DataA);
  assert(ReferenceB == DataB);
  assert(ReferenceC == DataC);

  return 0;
}
