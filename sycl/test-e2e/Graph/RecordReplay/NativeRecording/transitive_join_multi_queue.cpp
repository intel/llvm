// REQUIRES: level_zero_v2_adapter && (arch-intel_gpu_bmg_g21 || arch-intel_gpu_bmg_g31 || arch-intel_gpu_cri)
// REQUIRES: linux
// REQUIRES-INTEL-DRIVER: lin: 40074

// TODO: add minimum Windows driver when available

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test for native recording that a queue forked from the primary queue can be
// joined back transitively: Queue1 forks to Queue2 and Queue3, Queue3 joins
// Queue2, and only Queue2 joins Queue1.

#include "../../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  device Dev;
  context Ctx{Dev};

  queue Queue1{Ctx, Dev, {property::queue::in_order{}}};
  queue Queue2{Ctx, Dev, {property::queue::in_order{}}};
  queue Queue3{Ctx, Dev, {property::queue::in_order{}}};

  QueueStateVerifier verifier(Queue1, Queue2, Queue3);

  exp_ext::command_graph Graph{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};

  const size_t N = 1024;
  int *X = malloc_device<int>(N, Dev, Ctx);
  int *B = malloc_device<int>(N, Dev, Ctx);
  int *C = malloc_device<int>(N, Dev, Ctx);

  std::vector<int> HostX(N);
  std::iota(HostX.begin(), HostX.end(), 0);
  Queue1.copy(HostX.data(), X, N).wait();

  Graph.begin_recording(Queue1);
  verifier.verify(RECORDING, EXECUTING, EXECUTING);

  event Fork1 =
      Queue1.parallel_for(range<1>{N}, [=](item<1> Id) { X[Id] += 1; });
  event Fork2 = Queue1.ext_oneapi_submit_barrier();

  // Fork Queue1 to Queue2 and Queue3
  event Queue2Fork = Queue2.parallel_for(
      range<1>{N}, Fork1, [=](item<1> Id) { B[Id] = X[Id] * 2; });
  event Queue3Fork = Queue3.parallel_for(
      range<1>{N}, Fork2, [=](item<1> Id) { C[Id] = X[Id] + 3; });
  verifier.verify(RECORDING, RECORDING, RECORDING);

  // Join Queue3 to Queue2
  event Queue2Join = Queue2.parallel_for(range<1>{N}, Queue3Fork,
                                         [=](item<1> Id) { B[Id] += C[Id]; });

  // Join Queue2 to Queue1, which transitively joins Queue3 to Queue1
  Queue1.parallel_for(range<1>{N}, Queue2Join,
                      [=](item<1> Id) { X[Id] += B[Id]; });

  Graph.end_recording();
  verifier.verify(EXECUTING, EXECUTING, EXECUTING);

  auto ExecGraph = Graph.finalize();

  const size_t Iterations = 3;
  for (size_t I = 0; I < Iterations; I++) {
    Queue1.ext_oneapi_graph(ExecGraph);
  }
  Queue1.wait_and_throw();

  std::vector<int> OutX(N), OutB(N), OutC(N);
  Queue1.copy(X, OutX.data(), N);
  Queue1.copy(B, OutB.data(), N);
  Queue1.copy(C, OutC.data(), N);
  Queue1.wait_and_throw();

  for (size_t i = 0; i < N; i++) {
    int RefX = HostX[i];
    int RefB = 0;
    int RefC = 0;
    for (size_t I = 0; I < Iterations; I++) {
      RefX += 1;
      RefC = RefX + 3;
      RefB = RefX * 2 + RefC;
      RefX += RefB;
    }
    assert(check_value(i, RefX, OutX[i], "X"));
    assert(check_value(i, RefB, OutB[i], "B"));
    assert(check_value(i, RefC, OutC[i], "C"));
  }

  free(X, Ctx);
  free(B, Ctx);
  free(C, Ctx);

  return 0;
}
