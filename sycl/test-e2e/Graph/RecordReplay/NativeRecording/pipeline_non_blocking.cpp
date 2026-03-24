// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: env UR_L0_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s

// Test for native recording with pipelined graph execution. This test verifies
// that multiple graph executions can be batched with only a single host wait
// call at the end, demonstrating that all graph submissions are non-blocking.

#include "../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>

#include <iostream>

int main() {
  queue Queue{property::queue::in_order{}};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device(),
                               {exp_ext::property::graph::enable_native_recording{}}};

  const size_t N = 1024;
  int *Data = malloc_device<int>(N, Queue);

  QueueStateVerifier verifier(Queue);
  verifier.verify(EXECUTING);

  // Record graph with two kernels: add 3, then subtract 1
  Graph.begin_recording(Queue);
  verifier.verify(RECORDING);

  exp_ext::parallel_for(Queue, range<1>{N}, [=](id<1> idx) { Data[idx] += 3; });
  exp_ext::parallel_for(Queue, range<1>{N}, [=](id<1> idx) { Data[idx] -= 1; });

  Graph.end_recording(Queue);
  verifier.verify(EXECUTING);

  auto ExecutableGraph = Graph.finalize();

  // Initialize buffer, execute graph 20 times, copy results, then wait once
  exp_ext::memset(Queue, Data, 0, N * sizeof(int));

  std::cerr << "BEGIN_GRAPH_PIPELINE" << std::endl;
  for (int i = 0; i < 20; i++) {
    exp_ext::execute_graph(Queue, ExecutableGraph);
  }

  std::vector<int> HostData(N);
  exp_ext::memcpy(Queue, HostData.data(), Data, N * sizeof(int));
  Queue.wait();
  std::cerr << "END_GRAPH_PIPELINE" << std::endl;

  // Verify results: 20 iterations of (add 3, subtract 1) = 40
  const int Expected = 40;
  for (size_t i = 0; i < N; i++) {
    assert(check_value(i, Expected, HostData[i], "HostData"));
  }

  free(Data, Queue);
  return 0;
}

// Verify that there is exactly one host synchronization in the pipeline
// execution. The graph executions and memcpy should be batched asynchronously
// with only a single synchronization point at Queue.wait().
//
// At the L0 level, we should see exactly one zeCommandListHostSynchronize call
// (from Queue.wait()) and zero zeEventHostSynchronize calls in the execution
// region.
//
// CHECK-LABEL: BEGIN_GRAPH_PIPELINE
// CHECK-NOT: zeEventHostSynchronize(
// CHECK-COUNT-1: zeCommandListHostSynchronize(
// CHECK-NOT: zeCommandListHostSynchronize(
// CHECK: END_GRAPH_PIPELINE
