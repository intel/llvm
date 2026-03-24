// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

#include "../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue{property::queue::in_order{}};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device(),
                               {exp_ext::property::graph::enable_native_recording{}}};

  constexpr size_t N = 1024;

  Graph.begin_recording(Queue);

  // Test 1: async_malloc should throw during recording
  void *AsyncPtr = nullptr;
  if (!expectException(
          [&]() {
            Queue.submit([&](handler &CGH) {
              AsyncPtr = exp_ext::async_malloc(CGH, usm::alloc::device,
                                               N * sizeof(int));
            });
          },
          "async_malloc during native recording")) {
    Graph.end_recording();
    return 1;
  }

  // Test 2: async_free should also throw during recording
  // First allocate memory outside of recording
  Graph.end_recording();
  int *PreAllocatedPtr = malloc_device<int>(N, Queue);
  Graph.begin_recording(Queue);

  if (!expectException(
          [&]() {
            Queue.submit([&](handler &CGH) {
              exp_ext::async_free(CGH, PreAllocatedPtr);
            });
          },
          "async_free during native recording")) {
    Graph.end_recording();
    free(PreAllocatedPtr, Queue);
    return 1;
  }

  Graph.end_recording();
  free(PreAllocatedPtr, Queue);

  return 0;
}
