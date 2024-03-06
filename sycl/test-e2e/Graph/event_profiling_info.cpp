// REQUIRES: level_zero || cuda, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero && linux %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// This test checks the profiling of an event returned
// from graph submission with event::get_profiling_info().
// It first tests a graph made exclusively of memory operations,
// then tests a graph made of kernels.

#include "graph_common.hpp"

#define GRAPH_TESTS_VERBOSE_PRINT 0

#if GRAPH_TESTS_VERBOSE_PRINT
#include <chrono>
#endif

bool verifyProfiling(event Event) {
  auto Submit =
      Event.get_profiling_info<sycl::info::event_profiling::command_submit>();
  auto Start =
      Event.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto End =
      Event.get_profiling_info<sycl::info::event_profiling::command_end>();

#if GRAPH_TESTS_VERBOSE_PRINT
  std::cout << "Submit = " << Submit << std::endl;
  std::cout << "Start = " << Start << std::endl;
  std::cout << "End = " << End << " ( " << (End - Start) << " ) "
            << " => full ( " << (End - Submit) << " ) " << std::endl;
#endif

  assert((Submit && Start && End) && "Profiling information failed.");
  assert(Submit <= Start);
  assert(Start < End);

  bool Pass = sycl::info::event_command_status::complete ==
              Event.get_info<sycl::info::event::command_execution_status>();

  return Pass;
}

bool compareProfiling(event Event1, event Event2) {
  assert(Event1 != Event2);

  auto SubmitEvent1 =
      Event1.get_profiling_info<sycl::info::event_profiling::command_submit>();
  auto StartEvent1 =
      Event1.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto EndEvent1 =
      Event1.get_profiling_info<sycl::info::event_profiling::command_end>();
  assert((SubmitEvent1 && StartEvent1 && EndEvent1) &&
         "Profiling information failed.");

  auto SubmitEvent2 =
      Event2.get_profiling_info<sycl::info::event_profiling::command_submit>();
  auto StartEvent2 =
      Event2.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto EndEvent2 =
      Event2.get_profiling_info<sycl::info::event_profiling::command_end>();
  assert((SubmitEvent2 && StartEvent2 && EndEvent2) &&
         "Profiling information failed.");

  assert(SubmitEvent1 != SubmitEvent2);
  assert(StartEvent1 != StartEvent2);
  assert(EndEvent1 != EndEvent2);

  bool Pass1 = sycl::info::event_command_status::complete ==
               Event1.get_info<sycl::info::event::command_execution_status>();
  bool Pass2 = sycl::info::event_command_status::complete ==
               Event2.get_info<sycl::info::event::command_execution_status>();

  return (Pass1 && Pass2);
}

// The test checks that get_profiling_info waits for command associated with
// event to complete execution.
int main() {
  device Dev;
  queue Queue{Dev, {sycl::property::queue::enable_profiling()}};

  const size_t Size = 100000;
  int Data[Size] = {0};
  for (size_t I = 0; I < Size; ++I) {
    Data[I] = I;
  }
  int Values[Size] = {0};

  buffer<int, 1> BufferFrom(Data, range<1>(Size));
  buffer<int, 1> BufferTo(Values, range<1>(Size));

  buffer<int, 1> BufferA(Data, range<1>(Size));
  buffer<int, 1> BufferB(Values, range<1>(Size));
  buffer<int, 1> BufferC(Values, range<1>(Size));

  BufferFrom.set_write_back(false);
  BufferTo.set_write_back(false);
  BufferA.set_write_back(false);
  BufferB.set_write_back(false);
  BufferC.set_write_back(false);
  { // buffer copy
    exp_ext::command_graph CopyGraph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};
    CopyGraph.begin_recording(Queue);

    Queue.submit([&](sycl::handler &Cgh) {
      accessor<int, 1, access::mode::read, access::target::device> AccessorFrom(
          BufferFrom, Cgh, range<1>(Size));
      accessor<int, 1, access::mode::write, access::target::device> AccessorTo(
          BufferTo, Cgh, range<1>(Size));
      Cgh.copy(AccessorFrom, AccessorTo);
    });

    CopyGraph.end_recording(Queue);

    // kernel launch
    exp_ext::command_graph KernelGraph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};
    KernelGraph.begin_recording(Queue);

    run_kernels(Queue, Size, BufferA, BufferB, BufferC);

    KernelGraph.end_recording(Queue);

    // The `enable_profiling` property must be passed to the finalize function
    // in order to query profiling information.
    auto CopyGraphExec =
        CopyGraph.finalize(exp_ext::property::graph::enable_profiling{});
    auto KernelGraphExec =
        KernelGraph.finalize(exp_ext::property::graph::enable_profiling{});

    event CopyEvent, KernelEvent1, KernelEvent2;
    // Run graphs
#if GRAPH_TESTS_VERBOSE_PRINT
    auto StartCopyGraph = std::chrono::high_resolution_clock::now();
#endif
    CopyEvent = Queue.submit(
        [&](handler &CGH) { CGH.ext_oneapi_graph(CopyGraphExec); });
    Queue.wait_and_throw();
#if GRAPH_TESTS_VERBOSE_PRINT
    auto EndCopyGraph = std::chrono::high_resolution_clock::now();
    auto StartKernelSubmit1 = std::chrono::high_resolution_clock::now();
#endif
    KernelEvent1 = Queue.submit(
        [&](handler &CGH) { CGH.ext_oneapi_graph(KernelGraphExec); });
    Queue.wait_and_throw();
#if GRAPH_TESTS_VERBOSE_PRINT
    auto endKernelSubmit1 = std::chrono::high_resolution_clock::now();
    auto StartKernelSubmit2 = std::chrono::high_resolution_clock::now();
#endif
    KernelEvent2 = Queue.submit(
        [&](handler &CGH) { CGH.ext_oneapi_graph(KernelGraphExec); });
    Queue.wait_and_throw();
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

  host_accessor HostData(BufferTo);
  for (size_t I = 0; I < Size; ++I) {
    assert(HostData[I] == Values[I]);
  }

  return 0;
}
