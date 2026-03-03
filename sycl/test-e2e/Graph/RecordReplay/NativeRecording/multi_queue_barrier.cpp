// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test for native recording with fork-join using barriers. Barriers are used to
// record two independent streams of operations without any dependencies between
// each other apart from the queue recording transition and join.

#include "../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>

#include <algorithm>
#include <vector>

int main() {
  device Dev;
  context Ctx{Dev};

  queue Queue1{Ctx, Dev, {property::queue::in_order{}}};
  queue Queue2{Ctx, Dev, {property::queue::in_order{}}};

  const exp_ext::queue_state Recording = exp_ext::queue_state::recording;
  const exp_ext::queue_state Executing = exp_ext::queue_state::executing;

  auto assertQueueState = [&](exp_ext::queue_state ExpectedQ1,
                              exp_ext::queue_state ExpectedQ2) {
    assert(Queue1.ext_oneapi_get_state() == ExpectedQ1);
    assert(Queue2.ext_oneapi_get_state() == ExpectedQ2);
  };

  exp_ext::command_graph Graph{Ctx, Dev};

  const size_t N = 1024;

  int *VecA = malloc_device<int>(N, Dev, Ctx);
  int *VecB = malloc_device<int>(N, Dev, Ctx);

  assertQueueState(Executing, Executing);

  // 1) record on Queue1
  Graph.begin_recording(Queue1);
  assertQueueState(Recording, Executing);

  // 2) barrier (ext_oneapi_barrier) and recording transition on Queue2
  event ForkBarrier = Queue1.ext_oneapi_submit_barrier();
  exp_ext::partial_barrier(Queue2, {ForkBarrier});
  assertQueueState(Recording, Recording);

  // 3) two streams of independent kernels on Queue1 and Queue2
  exp_ext::parallel_for(Queue1, range<1>{N},
                        [=](item<1> idx) { VecA[idx] = 1; });
  exp_ext::parallel_for(Queue1, range<1>{N},
                        [=](item<1> idx) { VecA[idx] += 1; });
  exp_ext::parallel_for(Queue1, range<1>{N},
                        [=](item<1> idx) { VecA[idx] *= 2; });

  exp_ext::parallel_for(Queue2, range<1>{N},
                        [=](item<1> idx) { VecB[idx] = 2; });
  exp_ext::parallel_for(Queue2, range<1>{N},
                        [=](item<1> idx) { VecB[idx] *= 3; });
  exp_ext::parallel_for(Queue2, range<1>{N},
                        [=](item<1> idx) { VecB[idx] += 1; });

  // 4) join barrier
  event JoinBarrier = Queue2.ext_oneapi_submit_barrier();
  exp_ext::partial_barrier(Queue1, {JoinBarrier});

  Graph.end_recording();
  assertQueueState(Executing, Executing);

  // Finalize and execute the graph
  auto ExecutableGraph = Graph.finalize();

  exp_ext::execute_graph(Queue1, ExecutableGraph);
  Queue1.wait();

  // Verify results
  std::vector<int> HostVecA(N);
  std::vector<int> HostVecB(N);

  Queue1.memcpy(HostVecA.data(), VecA, N * sizeof(int)).wait();
  Queue1.memcpy(HostVecB.data(), VecB, N * sizeof(int)).wait();

  // VecA: 1 + 1 * 2 = 4
  assert(std::count(HostVecA.begin(), HostVecA.end(), 4) == N);

  // VecB: 2 * 3 + 1 = 7
  assert(std::count(HostVecB.begin(), HostVecB.end(), 7) == N);

  free(VecA, Ctx);
  free(VecB, Ctx);

  return 0;
}
