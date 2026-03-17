// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: %if level_zero %{env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK --input-file graph_native.dot %}

// Test for native recording with debug print graph functionality. SYCL graph
// does not control the output format, so we only verify high level details of
// the output and rely on validation tests in L0 graph

// CHECK: digraph
// CHECK: zeCommandListAppendMemoryFill
// CHECK: MyKernel
// CHECK: zeCommandListAppendMemoryCopy

#include "../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>

class MyKernel;

int main() {
  queue Queue{property::queue::in_order{}};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 128;

  int *DeviceData = malloc_device<int>(N, Queue);
  int *DeviceTemp = malloc_device<int>(N, Queue);

  Graph.begin_recording(Queue);

  exp_ext::memset(Queue, DeviceData, 0x42, N * sizeof(int));
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for<MyKernel>(
        range<1>{N}, [=](id<1> idx) { DeviceData[idx] = DeviceData[idx] + 1; });
  });
  exp_ext::memcpy(Queue, DeviceTemp, DeviceData, N * sizeof(int));

  Graph.end_recording(Queue);

  Graph.print_graph("graph_native.dot");

  free(DeviceData, Queue);
  free(DeviceTemp, Queue);

  return 0;
}
