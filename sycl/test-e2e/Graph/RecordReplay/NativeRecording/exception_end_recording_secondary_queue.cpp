// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests that end_recording(queue) throws for a queue other than the one
// begin_recording() was called on: feature_not_supported when the backend
// forked capture onto that queue, invalid when it is capturing into a
// different graph.

#include <iostream>

#include "../../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  device Dev;
  context Ctx{Dev};

  queue Queue1{Ctx, Dev, {property::queue::in_order{}}};
  queue Queue2{Ctx, Dev, {property::queue::in_order{}}};

  exp_ext::command_graph Graph{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};
  exp_ext::command_graph OtherGraph{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};

  bool Passed = true;

  Graph.begin_recording(Queue1);

  // Fork capture onto Queue2, which the SYCL runtime does not know about.
  auto ForkEvent = Queue1.ext_oneapi_submit_barrier();
  auto JoinEvent = Queue2.ext_oneapi_submit_barrier({ForkEvent});
  assert(Queue2.ext_oneapi_get_graph() == Graph);

  Passed &= expectException([&]() { Graph.end_recording(Queue2); },
                            "end_recording on a forked queue",
                            sycl::errc::feature_not_supported);

  // A queue capturing into another graph is a different error.
  Passed &= expectException([&]() { OtherGraph.end_recording(Queue1); },
                            "end_recording on a queue recording to another "
                            "graph",
                            sycl::errc::invalid);

  assert(Queue1.ext_oneapi_get_state() == exp_ext::queue_state::recording);
  assert(Queue2.ext_oneapi_get_state() == exp_ext::queue_state::recording);

  // Join the fork back so that capture can be ended.
  Queue1.ext_oneapi_submit_barrier({JoinEvent});
  Graph.end_recording(Queue1);

  if (!Passed) {
    std::cerr << "Expected exceptions from end_recording() were not thrown"
              << std::endl;
    return 1;
  }

  return 0;
}
