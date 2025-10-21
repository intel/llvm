// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test recording of handlerless SYCL queue APIs (memset, memcpy, memadvise,
// parallel_for/nd_launch, prefetch, single_task) inside a single function call
// while graph recording is active. All operations are USM-based and occur via
// eventless queue free functions or eventful queue shortcuts to bypass handler
// path. Recording is performed over a non-inlined function call.

#include "../graph_common.hpp"
#include <cstdint>
#include <cstring>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/sycl.hpp>

// noinline is important as we have previously caught functional issues with
// kernel argument capture only when the function being recorded is not inlined.
__attribute__((noinline)) void
recordHandlerLessOps(sycl::queue &Q, uint32_t *A, uint32_t *B, uint32_t *C,
                     uint32_t *D, uint32_t *E, size_t N, unsigned char Pattern,
                     uint32_t FillValue, bool InOrderQueue) {
  size_t WorkGroupSize = 16;
  sycl::nd_range<1> KernelRange{sycl::range<1>{N},
                                sycl::range<1>{WorkGroupSize}};
  auto DoubleKernelLambda = [=](sycl::nd_item<1> item) {
    const size_t i = item.get_global_linear_id();
    C[i] = B[i] * 2;
  };
  auto SingleTaskKernel = [=]() { C[0] = 999; };
  // Test eventless free functions with in-order queue and eventful shortcuts
  // with out-of-order queue.
  if (InOrderQueue) {
    exp_ext::memset(Q, A, Pattern, N * sizeof(uint32_t));
    exp_ext::fill(Q, D, FillValue, N);
    exp_ext::mem_advise(Q, D, N * sizeof(uint32_t), 0);
    exp_ext::copy(Q, D, E, N);
    exp_ext::memcpy(Q, B, A, N * sizeof(uint32_t));
    exp_ext::prefetch(Q, B, N * sizeof(uint32_t));
    exp_ext::nd_launch(Q, KernelRange, DoubleKernelLambda);
    exp_ext::single_task(Q, SingleTaskKernel);
  } else {
    auto e1 = Q.memset(A, Pattern, N * sizeof(uint32_t));
    auto e2 = Q.fill(D, FillValue, N);
    auto e3 = Q.mem_advise(D, N * sizeof(uint32_t), 0, e2);
    Q.copy(D, E, N, e3);
    auto e4 = Q.memcpy(B, A, N * sizeof(uint32_t), e1);
    auto e5 = Q.prefetch(B, N * sizeof(uint32_t), e4);
    auto e6 = Q.parallel_for(KernelRange, e5, DoubleKernelLambda);
    Q.single_task(e6, SingleTaskKernel);
  }
}

int main() {
  const size_t N = 64;
  const unsigned char Pattern = 42;
  const uint32_t FillValue = 7;
  auto getQueue = [](bool InOrder) {
    if (InOrder) {
      return sycl::queue{
          sycl::property_list{sycl::property::queue::in_order{}}};
    } else {
      return sycl::queue{};
    }
  };

  for (uint32_t i = 0; i <= 1; ++i) {
    const bool InOrderQueue = static_cast<bool>(i);
    sycl::queue Q = getQueue(InOrderQueue);
    uint32_t *A = sycl::malloc_shared<uint32_t>(N, Q);
    uint32_t *B = sycl::malloc_shared<uint32_t>(N, Q);
    uint32_t *C = sycl::malloc_shared<uint32_t>(N, Q);

    uint32_t *D = sycl::malloc_shared<uint32_t>(N, Q);
    uint32_t *E = sycl::malloc_shared<uint32_t>(N, Q);

    Q.memset(A, 0, N * sizeof(uint32_t));
    Q.memset(B, 0, N * sizeof(uint32_t));
    Q.memset(C, 0, N * sizeof(uint32_t));
    Q.memset(D, 0, N * sizeof(uint32_t));
    Q.memset(E, 0, N * sizeof(uint32_t));
    Q.wait_and_throw();

    exp_ext::command_graph Graph{Q.get_context(), Q.get_device()};
    Graph.begin_recording(Q);
    recordHandlerLessOps(Q, A, B, C, D, E, N, Pattern, FillValue, InOrderQueue);
    Graph.end_recording();

    auto Exec = Graph.finalize();
    Q.ext_oneapi_graph(Exec);
    Q.wait_and_throw();

    // Validate copy from D -> E
    for (size_t i = 0; i < N; ++i) {
      assert(check_value(i, FillValue, E[i], "E"));
    }

    // Validate final values in C
    assert(check_value(0, static_cast<uint32_t>(999), C[0], "C"));
    uint32_t DoublePatternUint = 0;
    std::memset(&DoublePatternUint, Pattern, sizeof(uint32_t));
    uint32_t DoublePatternUintDoubled = DoublePatternUint * 2;
    for (size_t i = 1; i < N; ++i) {
      assert(check_value(i, DoublePatternUintDoubled, C[i], "C"));
    }

    sycl::free(A, Q);
    sycl::free(B, Q);
    sycl::free(C, Q);
    sycl::free(D, Q);
    sycl::free(E, Q);
  }

  return 0;
}
