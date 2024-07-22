// REQUIRES: level_zero || cuda, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// This test checks the profiling of an event returned
// from graph submission with event::get_profiling_info().
// It first tests a graph made exclusively of memory operations,
// then tests a graph made of kernels. This test uses USM isntead of buffers to
// test the path with no implicit dependencies which bypasses the SYCL
// scheduler.
#define GRAPH_TESTS_VERBOSE_PRINT 0

#include "../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

// The test checks that get_profiling_info waits for command associated with
// event to complete execution.
int main() {
  device Dev;

  // Queue used for graph recording
  queue Queue{Dev};

  // Queue that will be used for execution
  queue ExecutionQueue{Queue.get_device(),
                       {sycl::property::queue::enable_profiling()}};

  const size_t Size = 100000;
  int Data[Size] = {0};
  for (size_t I = 0; I < Size; ++I) {
    Data[I] = I;
  }
  int Values[Size] = {0};
  int *PtrFrom = malloc_device<int>(Size, Queue);
  int *PtrTo = malloc_device<int>(Size, Queue);
  Queue.copy(Data, PtrFrom, Size);
  Queue.copy(Values, PtrTo, Size);

  int *PtrA = malloc_device<int>(Size, Queue);
  int *PtrB = malloc_device<int>(Size, Queue);
  int *PtrC = malloc_device<int>(Size, Queue);

  Queue.copy(Data, PtrA, Size);
  Queue.copy(Values, PtrB, Size);
  Queue.copy(Values, PtrC, Size);

  Queue.wait_and_throw();

  { // USM copy
    exp_ext::command_graph CopyGraph{Queue.get_context(), Queue.get_device()};
    CopyGraph.begin_recording(Queue);

    Queue.submit([&](sycl::handler &Cgh) {
      Cgh.memcpy(PtrTo, PtrFrom, Size * sizeof(int));
    });

    CopyGraph.end_recording(Queue);

    // kernel launch
    exp_ext::command_graph KernelGraph{Queue.get_context(), Queue.get_device()};
    KernelGraph.begin_recording(Queue);

    run_kernels_usm(Queue, Size, PtrA, PtrB, PtrC);

    KernelGraph.end_recording(Queue);

    auto CopyGraphExec =
        CopyGraph.finalize({exp_ext::property::graph::enable_profiling{}});
    auto KernelGraphExec =
        KernelGraph.finalize({exp_ext::property::graph::enable_profiling{}});

    event CopyEvent, KernelEvent1, KernelEvent2;
    // Run graphs
#if GRAPH_TESTS_VERBOSE_PRINT
    auto StartCopyGraph = std::chrono::high_resolution_clock::now();
#endif
    CopyEvent = ExecutionQueue.submit(
        [&](handler &CGH) { CGH.ext_oneapi_graph(CopyGraphExec); });
    ExecutionQueue.wait_and_throw();
#if GRAPH_TESTS_VERBOSE_PRINT
    auto EndCopyGraph = std::chrono::high_resolution_clock::now();
    auto StartKernelSubmit1 = std::chrono::high_resolution_clock::now();
#endif
    KernelEvent1 = ExecutionQueue.submit(
        [&](handler &CGH) { CGH.ext_oneapi_graph(KernelGraphExec); });
    ExecutionQueue.wait_and_throw();
#if GRAPH_TESTS_VERBOSE_PRINT
    auto endKernelSubmit1 = std::chrono::high_resolution_clock::now();
    auto StartKernelSubmit2 = std::chrono::high_resolution_clock::now();
#endif
    KernelEvent2 = ExecutionQueue.submit(
        [&](handler &CGH) { CGH.ext_oneapi_graph(KernelGraphExec); });
    ExecutionQueue.wait_and_throw();
#if GRAPH_TESTS_VERBOSE_PRINT
    auto endKernelSubmit2 = std::chrono::high_resolution_clock::now();

    double DelayCopy = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           EndCopyGraph - StartCopyGraph)
                           .count();
    std::cout << "Copy Graph delay (in ns) : " << DelayCopy << std::endl;
    double DelayKernel1 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              endKernelSubmit1 - StartKernelSubmit1)
                              .count();
    std::cout << "Kernel 1st Execution delay (in ns) : " << DelayKernel1
              << std::endl;
    double DelayKernel2 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              endKernelSubmit2 - StartKernelSubmit2)
                              .count();
    std::cout << "Kernel 2nd Execution delay (in ns) : " << DelayKernel2
              << std::endl;
#endif

    // Checks profiling times
    assert(verifyProfiling(CopyEvent) && verifyProfiling(KernelEvent1) &&
           verifyProfiling(KernelEvent2) &&
           compareProfiling(KernelEvent1, KernelEvent2));
  }

  std::vector<int> HostData(Size);
  Queue.copy(PtrTo, HostData.data(), Size).wait_and_throw();

  for (size_t I = 0; I < Size; ++I) {
    assert(HostData[I] == Data[I]);
  }

  return 0;
}
