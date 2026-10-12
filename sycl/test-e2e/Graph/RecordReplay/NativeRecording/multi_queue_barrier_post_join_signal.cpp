// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21
// REQUIRES: linux
// REQUIRES-INTEL-DRIVER: lin: 38591

// TODO: add minimum Windows driver when available

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test for native recording that unconsumed event signals are valid post-join

#include "../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>

int main() {
  device Dev;
  context Ctx{Dev};

  queue Queue1{Ctx, Dev, {property::queue::in_order{}}};
  queue Queue2{Ctx, Dev, {property::queue::in_order{}}};

  QueueStateVerifier verifier(Queue1, Queue2);

  exp_ext::command_graph Graph{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};

  verifier.verify(EXECUTING, EXECUTING);

  // 1) record on Queue1
  Graph.begin_recording(Queue1);
  verifier.verify(RECORDING, EXECUTING);

  // 2) barrier (ext_oneapi_barrier) and recording transition on Queue2
  event ForkBarrier = Queue1.ext_oneapi_submit_barrier();
  exp_ext::partial_barrier(Queue2, {ForkBarrier});
  verifier.verify(RECORDING, RECORDING);

  // 3) join barrier
  event JoinBarrier = Queue2.ext_oneapi_submit_barrier();
  exp_ext::partial_barrier(Queue1, {JoinBarrier});

  // 4) post join signal
  Queue2.ext_oneapi_submit_barrier();

  Graph.end_recording();
  verifier.verify(EXECUTING, EXECUTING);

  // We only need to check the graph finalizes without error
  [[maybe_unused]] auto ExecutableGraph = Graph.finalize();

  return 0;
}
