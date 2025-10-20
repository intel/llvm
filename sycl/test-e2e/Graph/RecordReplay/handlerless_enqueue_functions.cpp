// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test recording of handlerless SYCL queue APIs (memset, memcpy,
// parallel_for/nd_launch, prefetch, single_task) inside a single function call
// while graph recording is active. All operations are USM-based and occur via
// eventless queue free functions or eventful queue shortcuts to bypass handler
// path. Recording is performed over a non-inlined function call.

#include "../graph_common.hpp"
#include <cstring>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/sycl.hpp>

// Records a sequence of handlerless operations to exercise kernel argument
// capture in a function call:
// 1. memset A to a byte pattern
// 2. fill D with FillValue
// 3. copy D -> E
// 4. memcpy A -> B
// 5. prefetch B
// 6. kernel: C[i] = B[i] * 2 (DoubleKernel)
// 7. single_task: C[0] = 999
// noinline is important as we have caught functional issues only when the recording
// function is not inlined.
__attribute__((noinline)) void
record_handlerless_ops(sycl::queue &Q, uint32_t *A, uint32_t *B, uint32_t *C,
                       uint32_t *D, uint32_t *E, size_t N,
                       unsigned char Pattern, uint32_t FillValue,
                       bool UseFreeFunctions) {
  size_t WorkGroupSize = 16;
  sycl::nd_range<1> KernelRange{sycl::range<1>{N},
                                sycl::range<1>{WorkGroupSize}};
  auto DoubleKernelLambda = [=](sycl::nd_item<1> item) {
    const size_t i = item.get_global_linear_id();
    C[i] = B[i] * 2;
  };
  auto SingleTaskKernel = [=]() { C[0] = 999; };
  if (UseFreeFunctions) {
    exp_ext::memset(Q, A, Pattern, N * sizeof(uint32_t));
    exp_ext::fill(Q, D, FillValue, N);
    exp_ext::copy(Q, D, E, N);
    exp_ext::memcpy(Q, B, A, N * sizeof(uint32_t));
    exp_ext::prefetch(Q, B, N * sizeof(uint32_t));
    exp_ext::nd_launch(Q, KernelRange, DoubleKernelLambda);
    exp_ext::single_task(Q, SingleTaskKernel);
  } else {
    Q.memset(A, Pattern, N * sizeof(uint32_t));
    Q.fill(D, FillValue, N);
    Q.copy(D, E, N);
    Q.memcpy(B, A, N * sizeof(uint32_t));
    Q.prefetch(B, N * sizeof(uint32_t));
    Q.parallel_for(KernelRange, DoubleKernelLambda);
    Q.single_task(SingleTaskKernel);
  }
}

int main() {
  sycl::queue Q{sycl::property_list{sycl::property::queue::in_order{}}};
  const size_t N = 64;
  const unsigned char Pattern = 42;
  const uint32_t FillValue = 7;

  uint32_t *A = sycl::malloc_shared<uint32_t>(N, Q);
  uint32_t *B = sycl::malloc_shared<uint32_t>(N, Q);
  uint32_t *C = sycl::malloc_shared<uint32_t>(N, Q);

  uint32_t *D = sycl::malloc_shared<uint32_t>(N, Q);
  uint32_t *E = sycl::malloc_shared<uint32_t>(N, Q);

  for (uint32_t i = 0; i <= 1; ++i) {
    Q.memset(A, 0, N * sizeof(uint32_t));
    Q.memset(B, 0, N * sizeof(uint32_t));
    Q.memset(C, 0, N * sizeof(uint32_t));
    Q.memset(D, 0, N * sizeof(uint32_t));
    Q.memset(E, 0, N * sizeof(uint32_t));
    Q.wait_and_throw();

    exp_ext::command_graph Graph{Q.get_context(), Q.get_device()};
    // Begin recording, invoke function that issues handlerless ops, end
    // recording.
    Graph.begin_recording(Q);
    record_handlerless_ops(Q, A, B, C, D, E, N, Pattern, FillValue,
                           /*UseFreeFunctions=*/static_cast<bool>(i));
    Graph.end_recording();

    auto Exec = Graph.finalize();
    Q.ext_oneapi_graph(Exec);
    Q.wait_and_throw();

    // Validate results
    // C[0] overridden by single_task
    assert(check_value(0, static_cast<uint32_t>(999), C[0], "C"));
    uint32_t DoublePatternUint = 0;
    std::memset(&DoublePatternUint, Pattern, sizeof(uint32_t));
    uint32_t DoublePatternUintDoubled = DoublePatternUint * 2;
    for (size_t i = 1; i < N; ++i) {
      assert(check_value(i, DoublePatternUintDoubled, C[i], "C"));
    }

    // Validate fill & copy results
    for (size_t i = 0; i < N; ++i) {
      assert(check_value(i, FillValue, E[i], "E"));
    }
  }

  sycl::free(A, Q);
  sycl::free(B, Q);
  sycl::free(C, Q);
  sycl::free(D, Q);
  sycl::free(E, Q);

  return 0;
}
