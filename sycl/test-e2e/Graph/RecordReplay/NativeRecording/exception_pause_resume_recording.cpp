// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests that attempting to pause and resume recording is unsupported in native
// recording mode and throws an exception.

#include "../../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue{property::queue::in_order{}};

  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};

  const size_t N = 1024;
  int *Data = malloc_device<int>(N, Queue);

  QueueStateVerifier Verifier(Queue);
  Verifier.verify(EXECUTING);

  Graph.begin_recording(Queue);
  Verifier.verify(RECORDING);

  Queue.parallel_for(range<1>{N},
                     [=](id<1> idx) { Data[idx] = static_cast<int>(idx); });

  // Pause recording
  Graph.end_recording(Queue);
  Verifier.verify(EXECUTING);

  // Attempting to resume (begin_recording again on the same graph) must throw.
  const bool Passed =
      expectException([&]() { Graph.begin_recording(Queue); },
                      "begin_recording after end_recording on native graph",
                      sycl::errc::runtime);

  assert(Queue.ext_oneapi_get_state() == exp_ext::queue_state::executing);

  free(Data, Queue);

  if (!Passed) {
    std::cerr
        << "Expected an exception when resuming recording on a native graph"
        << std::endl;
    return 1;
  }

  return 0;
}
