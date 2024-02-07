// RUN: %{build} -fsycl-embed-ir -o %t.out
// RUN: env SYCL_RT_WARNING_LEVEL=1 %{run} %t.out 2>&1 | FileCheck %s
// REQUIRES: aspect-usm_shared_allocations
// Windows doesn't yet have full shutdown().
// UNSUPPORTED: ze_debug && windows

// Test fusion cancellation on an explicit memory operation on an USM pointer
// happening before complete_fusion.

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  constexpr size_t dataSize = 512;

  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};

  int *in1 = sycl::malloc_shared<int>(dataSize, q);
  int *in2 = sycl::malloc_shared<int>(dataSize, q);
  int *in3 = sycl::malloc_shared<int>(dataSize, q);
  int *tmp = sycl::malloc_shared<int>(dataSize, q);
  int *out = sycl::malloc_shared<int>(dataSize, q);
  int dst[dataSize];

  for (size_t i = 0; i < dataSize; ++i) {
    in1[i] = i * 2;
    in2[i] = i * 3;
    in3[i] = i * 4;
    tmp[i] = -1;
    out[i] = -1;
    dst[i] = -1;
  }

  ext::codeplay::experimental::fusion_wrapper fw{q};
  fw.start_fusion();

  assert(fw.is_in_fusion_mode() && "Queue should be in fusion mode");

  auto kernel1 = q.submit([&](handler &cgh) {
    cgh.parallel_for<class KernelOne>(
        dataSize, [=](id<1> i) { tmp[i] = in1[i] + in2[i]; });
  });

  auto kernel2 = q.submit([&](handler &cgh) {
    cgh.depends_on(kernel1);
    cgh.parallel_for<class KernelTwo>(
        dataSize, [=](id<1> i) { out[i] = tmp[i] * in3[i]; });
  });

  // This explicit copy operation has an explicit dependency on one of the
  // kernels and therefore requires synchronization. This should lead to
  // cancellation of the fusion.
  auto copyEvt = q.copy(tmp, dst, dataSize, kernel1);

  copyEvt.wait();

  assert(!fw.is_in_fusion_mode() &&
         "Queue should not be in fusion mode anymore");

  fw.complete_fusion({ext::codeplay::experimental::property::no_barriers{}});

  for (size_t i = 0; i < dataSize; ++i) {
    std::cout << out[i] << ", ";
  }
  std::cout << "\n";
  // Check the results
  for (size_t i = 0; i < dataSize; ++i) {
    assert(out[i] == (20 * i * i) && "Computation error");
    assert(dst[i] == (5 * i) && "Computation error");
  }

  sycl::free(in1, q);
  sycl::free(in2, q);
  sycl::free(in3, q);
  sycl::free(tmp, q);
  sycl::free(out, q);

  return 0;
}

// CHECK: WARNING: Aborting fusion because synchronization with one of the kernels in the fusion list was requested
