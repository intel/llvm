// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

#include "../../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  device Dev;
  context Ctx{Dev};

  // Create two in-order queues sharing the same device and context
  queue Queue1{Ctx, Dev, {property::queue::in_order{}}};
  queue Queue2{Ctx, Dev, {property::queue::in_order{}}};

  exp_ext::command_graph Graph{Ctx, Dev,
                               {exp_ext::property::graph::enable_native_recording{}}};

  constexpr size_t N = 1024;
  int *Data = malloc_device<int>(N, Dev, Ctx);

  Graph.begin_recording(Queue1);

  Queue1.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) { Data[idx] = idx; });
  });

  // Try to start recording on Queue2 while Queue1 is still recording
  const bool passed = expectException([&]() { Graph.begin_recording(Queue2); },
                       "begin_recording on second queue");

  assert(Queue1.ext_oneapi_get_state() == exp_ext::queue_state::recording);
  assert(Queue2.ext_oneapi_get_state() == exp_ext::queue_state::executing);
  
  Graph.end_recording(Queue1);
  free(Data, Ctx);

  if (!passed) {
    std::cerr << "Expected a thrown exception when starting recording twice" << std::endl;
    return 1;
  }

  return 0;
}
