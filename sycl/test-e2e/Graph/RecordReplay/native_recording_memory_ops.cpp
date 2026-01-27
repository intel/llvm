// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test for native recording with non-kernel memory operations using handlerless
// APIs: memcpy, memset, fill, copy

#include "../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>

int main() {
  // Create a queue with immediate command list property for native recording
  queue Queue{{property::queue::in_order{},
               ext::intel::property::queue::immediate_command_list{}}};

  // Create a graph - native recording is enabled via
  // SYCL_GRAPH_ENABLE_NATIVE_RECORDING environment variable
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 128;

  // Allocate two buffers
  int *Data1 = malloc_device<int>(2 * N, Queue);
  int *Data2 = malloc_device<int>(2 * N, Queue);

  // Host buffers for verification
  std::vector<int> HostData1(N);
  std::vector<int> HostData2(N);

  // Use queue recording mode to create the graph
  Graph.begin_recording(Queue);

  // Test memory operations on two buffers:
  // Data1: memset first half (byte-wise to 0x2A), D2D memcpy to second half and
  // D2H memcpy for verification Data2: fill first half, D2D copy to second half
  // and D2H copy for verification

  exp_ext::memset(Queue, Data1, 0x2A, N * sizeof(int));
  exp_ext::fill(Queue, Data2, 7, N);

  exp_ext::memcpy(Queue, Data1 + N, Data1, N * sizeof(int));
  exp_ext::memcpy(Queue, HostData1.data(), Data1 + N, N * sizeof(int));

  exp_ext::copy(Queue, Data2, Data2 + N, N);
  exp_ext::copy(Queue, Data2 + N, HostData2.data(), N);

  Graph.end_recording(Queue);

  // Finalize and execute the graph
  auto ExecutableGraph = Graph.finalize();

  exp_ext::execute_graph(Queue, ExecutableGraph);

  Queue.wait();

  for (size_t i = 0; i < N; i++) {
    assert(check_value(i, 0x2A2A2A2A, HostData1[i], "Data1"));
    assert(check_value(i, 7, HostData2[i], "Data2"));
  }

  free(Data1, Queue);
  free(Data2, Queue);

  return 0;
}
