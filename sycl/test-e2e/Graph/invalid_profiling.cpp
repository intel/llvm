// REQUIRES: level_zero || cuda, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out 2>&1
// RUN: %if ext_oneapi_level_zero %{env UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck --implicit-check-not=LEAK %s %}

// This test checks exception throwing if profiling an event returned
// from multi-partitions graph submission.

#define GRAPH_E2E_EXPLICIT

#include "graph_common.hpp"

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

  buffer<int, 1> BufferA(Data, range<1>(Size));
  buffer<int, 1> BufferB(Values, range<1>(Size));
  buffer<int, 1> BufferC(Values, range<1>(Size));

  BufferA.set_write_back(false);
  BufferB.set_write_back(false);
  BufferC.set_write_back(false);
  {
    // kernel launch
    exp_ext::command_graph KernelGraph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    // Add host_task to create a second partition.
    auto HostNode = add_node(KernelGraph, Queue,
                             [&](handler &CGH) { CGH.host_task([=]() {}); });

    auto Nodes = add_kernels(KernelGraph, Size, BufferA, BufferB, BufferC);

    KernelGraph.make_edge(HostNode, Nodes[0]);

    auto KernelGraphExec = KernelGraph.finalize();

    // Run graphs
    auto GraphEvent = Queue.submit(
        [&](handler &CGH) { CGH.ext_oneapi_graph(KernelGraphExec); });
    Queue.wait_and_throw();

    // get Submit timestamp should NOT work
    std::error_code ExceptionCode = make_error_code(sycl::errc::success);
    try {
      auto Submit = GraphEvent.get_profiling_info<
          sycl::info::event_profiling::command_submit>();
    } catch (sycl::exception &Exception) {
      ExceptionCode = Exception.code();
    }
    assert(ExceptionCode == sycl::errc::invalid);
    // get Start timestamp should NOT work
    ExceptionCode = make_error_code(sycl::errc::success);
    try {
      auto Start =
          GraphEvent
              .get_profiling_info<sycl::info::event_profiling::command_start>();
    } catch (sycl::exception &Exception) {
      ExceptionCode = Exception.code();
    }
    assert(ExceptionCode == sycl::errc::invalid);
    // get End timestamp should NOT work
    ExceptionCode = make_error_code(sycl::errc::success);
    try {
      auto End =
          GraphEvent
              .get_profiling_info<sycl::info::event_profiling::command_end>();
    } catch (sycl::exception &Exception) {
      ExceptionCode = Exception.code();
    }
    assert(ExceptionCode == sycl::errc::invalid);

    // get Submit timestamp should NOT work
    ExceptionCode = make_error_code(sycl::errc::success);
    try {
      auto NodeSubmit = GraphEvent.ext_oneapi_get_profiling_info<
          sycl::info::event_profiling::command_submit>(Nodes[0]);
    } catch (sycl::exception &Exception) {
      ExceptionCode = Exception.code();
    }
    assert(ExceptionCode == sycl::errc::invalid);
    // get Start timestamp should NOT work
    ExceptionCode = make_error_code(sycl::errc::success);
    try {
      auto NodeStart = GraphEvent.ext_oneapi_get_profiling_info<
          sycl::info::event_profiling::command_start>(Nodes[0]);
    } catch (sycl::exception &Exception) {
      ExceptionCode = Exception.code();
    }
    assert(ExceptionCode == sycl::errc::invalid);
    // get End timestamp should NOT work
    ExceptionCode = make_error_code(sycl::errc::success);
    try {
      auto NodeEnd = GraphEvent.ext_oneapi_get_profiling_info<
          sycl::info::event_profiling::command_end>(Nodes[0]);
    } catch (sycl::exception &Exception) {
      ExceptionCode = Exception.code();
    }
    assert(ExceptionCode == sycl::errc::invalid);

    exp_ext::command_graph SecondGraph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    // Add nodes
    auto SecondGraphNodes =
        add_kernels(SecondGraph, Size, BufferA, BufferB, BufferC);

    auto SecondGraphExec = SecondGraph.finalize();

    // Run graphs
    auto SecondGraphEvent = Queue.submit(
        [&](handler &CGH) { CGH.ext_oneapi_graph(SecondGraphExec); });
    Queue.wait_and_throw();

    // Query profiling of a node that do not belong to the executed graph.
    // get Submit timestamp should NOT work
    ExceptionCode = make_error_code(sycl::errc::success);
    try {
      auto NodeSubmit = SecondGraphEvent.ext_oneapi_get_profiling_info<
          sycl::info::event_profiling::command_submit>(Nodes[0]);
    } catch (sycl::exception &Exception) {
      ExceptionCode = Exception.code();
    }
    assert(ExceptionCode == sycl::errc::invalid);
    // get Start timestamp should NOT work
    ExceptionCode = make_error_code(sycl::errc::success);
    try {
      auto NodeStart = SecondGraphEvent.ext_oneapi_get_profiling_info<
          sycl::info::event_profiling::command_start>(Nodes[0]);
    } catch (sycl::exception &Exception) {
      ExceptionCode = Exception.code();
    }
    assert(ExceptionCode == sycl::errc::invalid);
    // get End timestamp should NOT work
    ExceptionCode = make_error_code(sycl::errc::success);
    try {
      auto NodeEnd = SecondGraphEvent.ext_oneapi_get_profiling_info<
          sycl::info::event_profiling::command_end>(Nodes[0]);
    } catch (sycl::exception &Exception) {
      ExceptionCode = Exception.code();
    }
    assert(ExceptionCode == sycl::errc::invalid);
  }

  return 0;
}
