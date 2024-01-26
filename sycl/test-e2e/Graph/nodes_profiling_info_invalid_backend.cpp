// RUN: %{build} -o %t.out
// RUN: %{run} %t.out 2>&1

// This test checks exception throwing when trying to get node profiling
// information on a non Level-Zero backend.

#define GRAPH_E2E_EXPLICIT

#include "graph_common.hpp"

int main() {
  device Dev;
  queue Queue{Dev,
              {sycl::ext::intel::property::queue::no_immediate_command_list{},
               sycl::property::queue::enable_profiling()}};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  // We do not excpeted to get exception with level-zero backend since node
  // profiling is supported by this backend.
  if (Dev.get_backend() == backend::ext_oneapi_level_zero) {
    return 0;
  }

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

    auto Nodes = add_kernels(KernelGraph, Size, BufferA, BufferB, BufferC);

    auto KernelGraphExec = KernelGraph.finalize();

    // Run graphs
    auto KernelEvent = Queue.submit(
        [&](handler &CGH) { CGH.ext_oneapi_graph(KernelGraphExec); });
    Queue.wait_and_throw();

    // get Submit timestamp should NOT work for backend other than level-zero
    std::error_code ExceptionCode = make_error_code(sycl::errc::success);
    try {
      auto NodeSubmit = KernelEvent.ext_oneapi_get_profiling_info<
          sycl::info::event_profiling::command_submit>(Nodes[0]);
    } catch (sycl::exception &Exception) {
      ExceptionCode = Exception.code();
    }
    assert(ExceptionCode == sycl::errc::invalid);
    // get Start timestamp should NOT work for backend other than level-zero
    ExceptionCode = make_error_code(sycl::errc::success);
    try {
      auto NodeStart = KernelEvent.ext_oneapi_get_profiling_info<
          sycl::info::event_profiling::command_start>(Nodes[0]);
    } catch (sycl::exception &Exception) {
      ExceptionCode = Exception.code();
    }
    assert(ExceptionCode == sycl::errc::invalid);
    // get End timestamp should NOT work for backend other than level-zero
    ExceptionCode = make_error_code(sycl::errc::success);
    try {
      auto NodeEnd = KernelEvent.ext_oneapi_get_profiling_info<
          sycl::info::event_profiling::command_end>(Nodes[0]);
    } catch (sycl::exception &Exception) {
      ExceptionCode = Exception.code();
    }
    assert(ExceptionCode == sycl::errc::invalid);
  }

  return 0;
}
