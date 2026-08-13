// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests that ending a recording with forked branches that were not rejoined
// throws errc::runtime. A fork is created when work branches off the recorded
// stream without being joined back before end_recording.

#include "../../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue1{property::queue::in_order{}};
  queue Queue2{Queue1.get_context(), Queue1.get_device(),
               property::queue::in_order{}};

  exp_ext::command_graph Graph{
      Queue1.get_context(),
      Queue1.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};

  constexpr size_t N = 1024;
  int *Data = malloc_device<int>(N, Queue1);

  Graph.begin_recording(Queue1);

  // Create a branch off the recorded stream that is never joined back before
  // ending the recording.
  auto ForkEvent = Queue1.parallel_for(
      sycl::range<1>{N}, [=](sycl::id<1> Idx) { Data[Idx] = Idx; });
  Queue2.submit([&](handler &CGH) {
    CGH.depends_on(ForkEvent);
    CGH.parallel_for(sycl::range<1>{N},
                     [=](sycl::id<1> Idx) { Data[Idx] += 1; });
  });

  if (!expectException([&]() { Graph.end_recording(); },
                       "end_recording with unjoined forks", errc::runtime)) {
    free(Data, Queue1);
    return 1;
  }

  free(Data, Queue1);

  return 0;
}
