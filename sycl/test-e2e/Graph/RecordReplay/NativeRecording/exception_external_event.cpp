// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests that mixing events across a graph recording boundary throws
// errc::runtime. Exercises the UR this external event path in
// two directions:
//   1. An event produced before begin_recording that is waited on while a
//      graph is being recorded (external event pulled into the graph).
//   2. An event recorded into the graph that is waited on outside of the graph
//      (internal graph event escaping the recording).

#include "../../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue{property::queue::in_order{}};

  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};

  // Event produced before recording began, i.e. external to the graph.
  auto ExternalEvent = Queue.single_task([]() {});

  Graph.begin_recording(Queue);

  // Waiting on an external event during recording fails in the runtime.
  if (!expectException([&]() { Queue.single_task(ExternalEvent, []() {}); },
                       "external event wait during graph recording",
                       errc::runtime)) {
    Graph.end_recording();
    return 1;
  }

  // Event recorded into the graph, i.e. internal to the graph.
  auto InternalEvent = Queue.single_task([]() {});

  Graph.end_recording();

  // Waiting on an internal graph event outside of the graph fails in the
  // runtime.
  if (!expectException([&]() { InternalEvent.wait(); },
                       "internal graph event wait outside graph recording",
                       errc::runtime)) {
    return 1;
  }

  return 0;
}
