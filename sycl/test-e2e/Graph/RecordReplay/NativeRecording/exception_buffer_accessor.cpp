// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

#include "../../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue{property::queue::in_order{}};

  constexpr size_t N = 1024;
  sycl::buffer<int> Buf{N};

  // Use assume_buffer_outlives_graph so the graph-level buffer check doesn't
  // fire before the native-recording check in the handler.
  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::enable_native_recording{},
       exp_ext::property::graph::assume_buffer_outlives_graph{}}};

  Graph.begin_recording(Queue);

  // Submitting a kernel that uses a buffer accessor should throw
  // errc::feature_not_supported in native recording mode.
  std::error_code ExceptionCode;
  try {
    Queue.submit([&](handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::write>(CGH);
      CGH.parallel_for(sycl::range<1>{N},
                       [=](sycl::id<1> Idx) { Acc[Idx] = 0; });
    });
    std::cerr << "ERROR: Expected exception was not thrown for buffer accessor "
                 "in native recording mode"
              << std::endl;
    Graph.end_recording();
    return 1;
  } catch (const sycl::exception &e) {
    ExceptionCode = e.code();
  }
  assert(ExceptionCode == sycl::errc::feature_not_supported);

  Graph.end_recording();

  return 0;
}
