// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test for SYCL_GRAPH_ENABLE_NATIVE_RECORDING with multi-queue dot product
// Assesses event dependencies to, from, and within a native recording graph

#include "../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>

int main() {
  // Create context and device
  device Dev;
  context Ctx{Dev};

  // Create two in-order queues sharing the same device and context
  queue Queue1{Ctx,
               Dev,
               {property::queue::in_order{},
                ext::intel::property::queue::immediate_command_list{}}};
  queue Queue2{Ctx,
               Dev,
               {property::queue::in_order{},
                ext::intel::property::queue::immediate_command_list{}}};

  const exp_ext::queue_state Recording = exp_ext::queue_state::recording;
  const exp_ext::queue_state Executing = exp_ext::queue_state::executing;

  auto assertQueueState = [&](exp_ext::queue_state ExpectedQ1,
                              exp_ext::queue_state ExpectedQ2) {
    assert(Queue1.ext_oneapi_get_state() == ExpectedQ1);
    assert(Queue2.ext_oneapi_get_state() == ExpectedQ2);
  };

  // Create a graph - native recording is enabled via
  // SYCL_GRAPH_ENABLE_NATIVE_RECORDING environment variable
  exp_ext::command_graph Graph{Ctx, Dev};

  const size_t N = 1024;

  // Allocate input vectors and partial result buffers
  int *VecA = malloc_device<int>(N, Dev, Ctx);
  int *VecB = malloc_device<int>(N, Dev, Ctx);
  int *PartialResult1 = malloc_device<int>(1, Dev, Ctx);
  int *PartialResult2 = malloc_device<int>(1, Dev, Ctx);
  int *FinalResult = malloc_device<int>(1, Dev, Ctx);

  assertQueueState(Executing, Executing);

  // Begin graph recording on Queue1 only
  Graph.begin_recording(Queue1);
  assertQueueState(Recording, Executing);

  // Transform VecA
  event Fork =
      Queue1.parallel_for(range<1>{N}, [=](item<1> idx) { VecA[idx] *= 2; });

  // Record partial dot product on first half with Queue2, transitioning to
  // recording (fork)
  event Join = Queue2.single_task({Fork}, [=]() {
    int sum = 0;
    for (size_t i = 0; i < N / 2; i++) {
      sum += VecA[i] * VecB[i];
    }
    PartialResult1[0] = sum;
  });
  assertQueueState(Recording, Recording);

  // Record partial dot product on second half (Queue1)
  exp_ext::single_task(Queue1, [=]() {
    int sum = 0;
    for (size_t i = N / 2; i < N; i++) {
      sum += VecA[i] * VecB[i];
    }
    PartialResult2[0] = sum;
  });

  // Record final reduction kernel with dependency on Queue2 event
  Queue1.single_task({Join}, [=]() {
    FinalResult[0] = PartialResult1[0] + PartialResult2[0];
  });

  Graph.end_recording();
  assertQueueState(Executing, Executing);

  // Finalize and execute the graph
  auto ExecutableGraph = Graph.finalize();

  // Initialize input vector outside of graph. Use Queue2 to be able to test
  // graph dependent event
  event InitEvent = Queue2.parallel_for(range<1>{N}, [=](item<1> idx) {
    VecA[idx] = static_cast<int>(idx);
    VecB[idx] = static_cast<int>(idx) + 1;
  });

  auto GraphEvent = Queue1.ext_oneapi_graph(ExecutableGraph, {InitEvent});

  // Wait for graph completion
  GraphEvent.wait();

  // Verify result
  int HostResult = 0;
  Queue1.memcpy(&HostResult, FinalResult, sizeof(int));
  Queue1.wait();

  // Compute expected result
  int Expected = 0;
  for (int i = 0; i < N; i++) {
    Expected += 2 * i * (i + 1);
  }

  assert(check_value(0, Expected, HostResult, "DotProduct"));

  free(VecA, Ctx);
  free(VecB, Ctx);
  free(PartialResult1, Ctx);
  free(PartialResult2, Ctx);
  free(FinalResult, Ctx);

  return 0;
}
