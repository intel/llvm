// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// REQUIRES: ocloc && level_zero

// Tests updating a raw_kernel_arg that was built as a pointer argument, which
// has to rebind the pointer rather than the bytes one is made of.

#include "../graph_common.hpp"

auto constexpr CLSource = R"===(
__kernel void RawArgKernel(int scalar, __global int *out) {
  size_t id = get_global_id(0);
  out[id] = id + scalar;
}
)===";

int main() {
  queue Queue{};

  auto SourceKB =
      sycl::ext::oneapi::experimental::create_kernel_bundle_from_source(
          Queue.get_context(),
          sycl::ext::oneapi::experimental::source_language::opencl, CLSource);
  auto ExecKB = sycl::ext::oneapi::experimental::build(SourceKB);

  exp_ext::command_graph Graph{Queue};

  const size_t N = 1024;
  int32_t *PtrA = malloc_device<int32_t>(N, Queue);
  int32_t *PtrB = malloc_device<int32_t>(N, Queue);
  Queue.memset(PtrA, 0, N * sizeof(int32_t));
  Queue.memset(PtrB, 0, N * sizeof(int32_t));
  Queue.wait();

  int32_t Scalar = 42;
  exp_ext::raw_kernel_arg RawScalar(&Scalar, sizeof(int32_t));
  exp_ext::raw_kernel_arg RawPtrA(&PtrA, exp_ext::pointer_arg);
  exp_ext::raw_kernel_arg RawPtrB(&PtrB, exp_ext::pointer_arg);

  exp_ext::dynamic_parameter ScalarParam(RawScalar);
  exp_ext::dynamic_parameter PtrParam(RawPtrA);

  auto KernelNode = Graph.add([&](handler &cgh) {
    cgh.set_arg(0, ScalarParam);
    cgh.set_arg(1, PtrParam);
    cgh.parallel_for(sycl::range<1>{N},
                     ExecKB.ext_oneapi_get_kernel("RawArgKernel"));
  });

  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  // PtrA is the one the pointer argument was built from.
  Queue.ext_oneapi_graph(ExecGraph).wait();

  std::vector<int> HostDataA(N);
  std::vector<int> HostDataB(N);
  Queue.copy(PtrA, HostDataA.data(), N);
  Queue.copy(PtrB, HostDataB.data(), N);
  Queue.wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostDataA[i] == static_cast<int>(i + Scalar));
    assert(HostDataB[i] == 0);
  }

  // Rebind the pointer argument to the other allocation.
  PtrParam.update(RawPtrB);
  ExecGraph.update(KernelNode);
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(PtrA, HostDataA.data(), N).wait();
  Queue.copy(PtrB, HostDataB.data(), N).wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostDataA[i] == static_cast<int>(i + Scalar));
    assert(HostDataB[i] == static_cast<int>(i + Scalar));
  }

  sycl::free(PtrA, Queue);
  sycl::free(PtrB, Queue);
  return 0;
}
