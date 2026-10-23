// REQUIRES: level_zero_v2_adapter && (arch-intel_gpu_bmg_g21 || arch-intel_gpu_bmg_g31 || arch-intel_gpu_cri)

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Submits several executable graphs finalized from one native recording graph
// round-robin across queues so they may run concurrently. All instances share
// the same USM, so the graph only reads inputs and accumulates with atomics.

#include "../../graph_common.hpp"

#include <sycl/atomic_ref.hpp>
#include <sycl/properties/all_properties.hpp>

template <typename T>
using device_atomic_ref =
    atomic_ref<T, memory_order::relaxed, memory_scope::device,
               access::address_space::global_space>;

int main() {
  device Dev;
  context Ctx{Dev};

  constexpr size_t NumQueues = 4;
  constexpr size_t NumExecGraphs = 2 * NumQueues;
  std::vector<queue> Queues;
  for (size_t i = 0; i < NumQueues; i++) {
    Queues.emplace_back(Ctx, Dev, property::queue::in_order{});
  }

  exp_ext::command_graph Graph{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};

  int *InputA = malloc_device<int>(Size, Dev, Ctx);
  int *InputB = malloc_device<int>(Size, Dev, Ctx);
  int *Products = malloc_device<int>(Size, Dev, Ctx);
  int *Sum = malloc_device<int>(1, Dev, Ctx);

  std::vector<int> HostA(Size), HostB(Size);
  std::iota(HostA.begin(), HostA.end(), 0);
  std::iota(HostB.begin(), HostB.end(), 1);

  queue &RecordQueue = Queues[0];
  RecordQueue.copy(HostA.data(), InputA, Size);
  RecordQueue.copy(HostB.data(), InputB, Size);
  RecordQueue.fill(Products, 0, Size);
  RecordQueue.fill(Sum, 0, 1);
  RecordQueue.wait_and_throw();

  Graph.begin_recording(RecordQueue);

  RecordQueue.parallel_for(range<1>{Size}, [=](id<1> idx) {
    device_atomic_ref<int>{Products[idx]} += InputA[idx] * InputB[idx];
  });

  RecordQueue.parallel_for(range<1>{Size}, [=](id<1> idx) {
    device_atomic_ref<int>{Sum[0]} += InputA[idx];
  });

  Graph.end_recording(RecordQueue);

  std::vector<exp_ext::command_graph<exp_ext::graph_state::executable>>
      ExecGraphs;
  for (size_t i = 0; i < NumExecGraphs; i++) {
    ExecGraphs.push_back(Graph.finalize());
  }

  for (unsigned n = 0; n < Iterations; n++) {
    for (size_t i = 0; i < NumExecGraphs; i++) {
      Queues[i % NumQueues].ext_oneapi_graph(ExecGraphs[i]);
    }
  }

  for (queue &Q : Queues) {
    Q.wait_and_throw();
  }

  std::vector<int> HostProducts(Size);
  int HostSum = 0;
  RecordQueue.copy(Products, HostProducts.data(), Size);
  RecordQueue.copy(Sum, &HostSum, 1);
  RecordQueue.wait_and_throw();

  const int TotalExecutions = NumExecGraphs * Iterations;
  int ExpectedSum = 0;
  for (size_t i = 0; i < Size; i++) {
    int Expected = TotalExecutions * HostA[i] * HostB[i];
    assert(check_value(i, Expected, HostProducts[i], "Products"));
    ExpectedSum += TotalExecutions * HostA[i];
  }
  assert(check_value(0, ExpectedSum, HostSum, "Sum"));

  free(InputA, Ctx);
  free(InputB, Ctx);
  free(Products, Ctx);
  free(Sum, Ctx);

  return 0;
}
