// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out 2>&1
// RUN: %if ext_oneapi_level_zero %{env UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck --implicit-check-not=LEAK %s %}

// This test checks the profiling of an event returned
// from graph submission with event::ext_oneapi_get_profiling_info().
// It first tests a graph made exclusively of memory operations,
// then tests a graph made of kernels.
// The second run is to check that there are no leaks reported with the embedded
// UR_L0_LEAKS_DEBUG testing capability.

#include "graph_common.hpp"

#define GRAPH_TESTS_VERBOSE_PRINT 0

#if GRAPH_TESTS_VERBOSE_PRINT
#include <chrono>
#endif

bool verifyProfiling(event &Event, exp_ext::node &Node) {
  auto Submit = Event.ext_oneapi_get_profiling_info<
      sycl::info::event_profiling::command_submit>(Node);
  auto Start = Event.ext_oneapi_get_profiling_info<
      sycl::info::event_profiling::command_start>(Node);
  auto End = Event.ext_oneapi_get_profiling_info<
      sycl::info::event_profiling::command_end>(Node);

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

int main() {
  device Dev;
  queue Queue{Dev,
              {sycl::ext::intel::property::queue::no_immediate_command_list{},
               sycl::property::queue::enable_profiling()}};

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

    auto NodeCopy = CopyGraph.add([&](sycl::handler &Cgh) {
      accessor<int, 1, access::mode::read, access::target::device> AccessorFrom(
          BufferFrom, Cgh, range<1>(Size));
      accessor<int, 1, access::mode::write, access::target::device> AccessorTo(
          BufferTo, Cgh, range<1>(Size));
      Cgh.copy(AccessorFrom, AccessorTo);
    });

    // kernel launch
    exp_ext::command_graph KernelGraph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    auto Nodes = add_kernels(KernelGraph, Size, BufferA, BufferB, BufferC);

    auto CopyGraphExec = CopyGraph.finalize();
    auto KernelGraphExec = KernelGraph.finalize();

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

    // The copy graph is only made of a single node so all graph timestamps must
    // be the same that node timestamps.
    auto CopySubmit =
        CopyEvent
            .get_profiling_info<sycl::info::event_profiling::command_submit>();
    auto CopyStart =
        CopyEvent
            .get_profiling_info<sycl::info::event_profiling::command_start>();
    auto CopyEnd =
        CopyEvent
            .get_profiling_info<sycl::info::event_profiling::command_end>();

    assert(verifyProfiling(CopyEvent, NodeCopy));
    auto NodeCopySubmit = CopyEvent.ext_oneapi_get_profiling_info<
        sycl::info::event_profiling::command_submit>(NodeCopy);
    auto NodeCopyStart = CopyEvent.ext_oneapi_get_profiling_info<
        sycl::info::event_profiling::command_start>(NodeCopy);
    auto NodeCopyEnd = CopyEvent.ext_oneapi_get_profiling_info<
        sycl::info::event_profiling::command_end>(NodeCopy);

    assert((CopySubmit == NodeCopySubmit) && "Copy submit times differ");
    assert((CopyStart == NodeCopyStart) && "Copy start times differ");
    assert((CopyEnd == NodeCopyEnd) && "Copy end times differ");

    // Check first execution

    auto Submit =
        KernelEvent1
            .get_profiling_info<sycl::info::event_profiling::command_submit>();
    auto Start =
        KernelEvent1
            .get_profiling_info<sycl::info::event_profiling::command_start>();
    auto End =
        KernelEvent1
            .get_profiling_info<sycl::info::event_profiling::command_end>();

    std::vector<uint64_t> NodeSubmit(Nodes.size());
    std::vector<uint64_t> NodeStart(Nodes.size());
    std::vector<uint64_t> NodeEnd(Nodes.size());
    uint64_t GlobalRuntime;
    for (int i = 0; i < Nodes.size(); i++) {

      assert(verifyProfiling(KernelEvent1, Nodes[i]));

      NodeSubmit[i] = KernelEvent1.ext_oneapi_get_profiling_info<
          sycl::info::event_profiling::command_submit>(Nodes[i]);
      NodeStart[i] = KernelEvent1.ext_oneapi_get_profiling_info<
          sycl::info::event_profiling::command_start>(Nodes[i]);
      NodeEnd[i] = KernelEvent1.ext_oneapi_get_profiling_info<
          sycl::info::event_profiling::command_end>(Nodes[i]);

      GlobalRuntime += (NodeEnd[i] - NodeStart[i]);
    }

    // submit time should be all the same
    assert((Submit == NodeSubmit[0]) && "Submit times differ");
    assert((Submit == NodeSubmit[1]) && "Submit times differ");
    assert((Submit == NodeSubmit[2]) && "Submit times differ");
    assert((Submit == NodeSubmit[3]) && "Submit times differ");

    // check start timestamps order
    assert((Start == NodeStart[0]) &&
           "Graph timestamp and First node timestamp differ");
    assert(NodeStart[0] < NodeStart[1]);
    assert(NodeEnd[0] <= NodeStart[1]);
    assert(NodeStart[0] < NodeStart[2]);
    assert(NodeEnd[0] <= NodeStart[2]);
    assert(NodeStart[1] < NodeStart[3]);
    assert(NodeEnd[1] <= NodeStart[3]);
    assert(NodeStart[2] < NodeStart[3]);
    assert(NodeEnd[2] <= NodeStart[3]);
    assert((End == NodeEnd[3]) &&
           "Graph timestamp and last node timestamp differ");

#if GRAPH_TESTS_VERBOSE_PRINT
    std::cout << "Global Node Runtime = " << GlobalRuntime << std::endl;
    std::cout << "Global Node Runtime timestamp based = "
              << (NodeEnd[3] - NodeStart[0]) << std::endl;
    std::cout << "Graph runtime = " << (End - Start) << std::endl;
#endif

    assert(((NodeEnd[3] - NodeStart[0]) == (End - Start)) &&
           "Global Runtime differ");
    // The graph node addtional time to handle sychornization. So the sum of
    // node runtimes should be slightly lower than the whole graph runtime.
    assert(GlobalRuntime <= (End - Start));

    // Checks profiling times for two executions of the same graph.
    assert(compareProfiling(KernelEvent1, KernelEvent2));

    // Check second execution
    auto Submit2 =
        KernelEvent2
            .get_profiling_info<sycl::info::event_profiling::command_submit>();
    auto Start2 =
        KernelEvent2
            .get_profiling_info<sycl::info::event_profiling::command_start>();
    auto End2 =
        KernelEvent2
            .get_profiling_info<sycl::info::event_profiling::command_end>();

    std::vector<uint64_t> NodeSubmit2(Nodes.size());
    std::vector<uint64_t> NodeStart2(Nodes.size());
    std::vector<uint64_t> NodeEnd2(Nodes.size());
    uint64_t GlobalRuntime2;
    for (int i = 0; i < Nodes.size(); i++) {

      assert(verifyProfiling(KernelEvent2, Nodes[i]));

      NodeSubmit2[i] = KernelEvent2.ext_oneapi_get_profiling_info<
          sycl::info::event_profiling::command_submit>(Nodes[i]);
      NodeStart2[i] = KernelEvent2.ext_oneapi_get_profiling_info<
          sycl::info::event_profiling::command_start>(Nodes[i]);
      NodeEnd2[i] = KernelEvent2.ext_oneapi_get_profiling_info<
          sycl::info::event_profiling::command_end>(Nodes[i]);

      GlobalRuntime2 += (NodeEnd2[i] - NodeStart2[i]);
    }

    // submit time should be all the same
    assert((Submit2 == NodeSubmit2[0]) && "Submit times differ");
    assert((Submit2 == NodeSubmit2[1]) && "Submit times differ");
    assert((Submit2 == NodeSubmit2[2]) && "Submit times differ");
    assert((Submit2 == NodeSubmit2[3]) && "Submit times differ");

    // check start timestamps order
    assert((Start2 == NodeStart2[0]) &&
           "Graph timestamp and First node timestamp differ");
    assert(NodeStart2[0] < NodeStart2[1]);
    assert(NodeEnd2[0] <= NodeStart2[1]);
    assert(NodeStart2[0] < NodeStart2[2]);
    assert(NodeEnd2[0] <= NodeStart2[2]);
    assert(NodeStart2[1] < NodeStart2[3]);
    assert(NodeEnd2[1] <= NodeStart2[3]);
    assert(NodeStart2[2] < NodeStart2[3]);
    assert(NodeEnd2[2] <= NodeStart2[3]);
    assert((End2 == NodeEnd2[3]) &&
           "Graph timestamp and last node timestamp differ");

#if GRAPH_TESTS_VERBOSE_PRINT
    std::cout << "Global Node Runtime (second run) = " << GlobalRuntime2
              << std::endl;
    std::cout << "Global Node Runtime timestamp based (second run) = "
              << (NodeEnd2[3] - NodeStart2[0]) << std::endl;
    std::cout << "Graph runtime (second run) = " << (End2 - Start2)
              << std::endl;
#endif

    assert(((NodeEnd2[3] - NodeStart2[0]) == (End2 - Start2)) &&
           "Global Runtime differ");
    // The graph node addtional time to handle sychornization. So the sum of
    // node runtimes should be slightly lower than the whole graph runtime.
    assert(GlobalRuntime2 <= (End2 - Start2));
  }

  host_accessor HostData(BufferTo);
  for (size_t I = 0; I < Size; ++I) {
    assert(HostData[I] == Values[I]);
  }

  return 0;
}
