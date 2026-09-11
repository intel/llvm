// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21
// REQUIRES: linux
// REQUIRES-INTEL-DRIVER: lin: 39938

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests that mixing events across a graph recording boundary throws
// errc::runtime. Exercises the UR external event path in
// two directions:
//   1. An event produced before begin_recording that is waited on while a
//      graph is being recorded (external event pulled into the graph).
//   2. An event recorded into the graph that is waited on outside of the graph
//      (internal graph event escaping the recording).

#include "../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue{property::queue::in_order{}};

  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};

  // Event produced before recording began, i.e. external to the graph.
  auto ExternalEvent = Queue.single_task([]() {});
  auto ReusableEvent = exp_ext::make_event(Queue.get_context());
  exp_ext::enqueue_signal_event(Queue, ReusableEvent);

  Graph.begin_recording(Queue);

  // Waiting on an external event during recording fails in the runtime.
  bool Success = expectException(
      [&] { Queue.single_task(ExternalEvent, []() {}); },
      "external event dependency of kernel recorded into graph", errc::runtime);
  Success &= expectException(
      [&] { exp_ext::enqueue_wait_event(Queue, ReusableEvent); },
      "external reusable event wait during graph recording", errc::runtime);

  // Event recorded into the graph, i.e. internal to the graph.
  auto InternalEvent1 = Queue.single_task([]() {});
  auto InternalEvent2 = Queue.ext_oneapi_submit_barrier();
  exp_ext::enqueue_signal_event(Queue, ReusableEvent);

  Graph.end_recording();

  // Waiting on an internal graph event outside of the graph fails in the
  // runtime.
  Success &= expectException(
      [&] { InternalEvent1.wait(); },
      "internal graph event wait outside graph recording", errc::runtime);
  Success &= expectException(
      [&] { Queue.ext_oneapi_submit_barrier({InternalEvent2}); },
      "internal graph event dependency of barrier outside graph recording",
      errc::runtime);
  Success &= expectException(
      [&] { exp_ext::enqueue_wait_event(Queue, ReusableEvent); },
      "internal reusable event wait outside graph recording", errc::runtime);

  return Success ? 0 : 1;
}
