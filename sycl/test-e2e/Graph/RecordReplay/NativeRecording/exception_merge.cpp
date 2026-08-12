// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests that an illegal attempt to merge two graph recordings throws
// errc::runtime. A merge attempt arises when work already being captured
// by one graph is pulled into the capture of a second graph via a cross-graph
// dependency.

#include "../../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  queue QueueA{property::queue::in_order{}};
  queue QueueB{QueueA.get_context(), QueueA.get_device(),
               property::queue::in_order{}};

  exp_ext::command_graph GraphA{
      QueueA.get_context(),
      QueueA.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};
  exp_ext::command_graph GraphB{
      QueueB.get_context(),
      QueueB.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};

  constexpr size_t N = 1024;
  int *Data = malloc_device<int>(N, QueueA);

  GraphA.begin_recording(QueueA);
  GraphB.begin_recording(QueueB);

  auto EventA = QueueA.parallel_for(sycl::range<1>{N},
                                    [=](sycl::id<1> Idx) { Data[Idx] = Idx; });

  // Recording work on QueueB that depends on QueueA's in-progress capture
  // would merge the two separate recordings, which is illegal.
  if (!expectException(
          [&]() {
            QueueB.submit([&](handler &CGH) {
              CGH.depends_on(EventA);
              CGH.parallel_for(sycl::range<1>{N},
                               [=](sycl::id<1> Idx) { Data[Idx] += 1; });
            });
          },
          "merging two graph recordings", errc::runtime)) {
    GraphA.end_recording(QueueA);
    GraphB.end_recording(QueueB);
    free(Data, QueueA);
    return 1;
  }

  GraphA.end_recording(QueueA);
  GraphB.end_recording(QueueB);
  free(Data, QueueA);

  return 0;
}
