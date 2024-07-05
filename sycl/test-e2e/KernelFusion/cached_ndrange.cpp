// RUN: %{build} %{embed-ir} -O2 -o %t.out
// RUN: env SYCL_RT_WARNING_LEVEL=1 %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not "COMPUTATION ERROR"
// UNSUPPORTED: hip

// Test caching for JIT fused kernels. Also test for debug messages being
// printed when SYCL_RT_WARNING_LEVEL=1.

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>
#include <sycl/properties/all_properties.hpp>

using namespace sycl;

constexpr size_t dataSize = 512;

void performFusion(queue &q, range<1> globalSize) {
  std::array<int, dataSize> in1, tmp, out;

  for (size_t i = 0; i < dataSize; ++i) {
    in1[i] = i * 3;
    tmp[i] = -1;
    out[i] = -1;
  }

  {
    buffer<int> bIn1{in1.data(), globalSize};
    buffer<int> bTmp{tmp.data(), globalSize};
    buffer<int> bOut{out.data(), globalSize};

    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();

    assert(fw.is_in_fusion_mode() && "Queue should be in fusion mode");

    q.submit([&](handler &cgh) {
      auto accIn1 = bIn1.get_access(cgh);
      accessor<int> accTmp = bTmp.get_access(cgh);
      cgh.parallel_for<class KernelOne>(
          globalSize, [=](id<1> i) { accTmp[i] = accIn1[i] + 7; });
    });

    q.submit([&](handler &cgh) {
      accessor<int> accTmp = bTmp.get_access(cgh);
      auto accOut = bOut.get_access(cgh);
      cgh.parallel_for<class KernelTwo>(
          globalSize, [=](id<1> i) { accOut[i] = accTmp[i] * 11; });
    });

    fw.complete_fusion();

    assert(!fw.is_in_fusion_mode() &&
           "Queue should not be in fusion mode anymore");
  }

  // Check the results
  size_t numErrors = 0;
  for (size_t i = 0; i < dataSize; ++i) {
    if (i < globalSize.size() && out[i] != ((i * 3 + 7) * 11)) {
      ++numErrors;
    }
  }
  if (numErrors) {
    std::cout << "COMPUTATION ERROR\n";
  }
}

int main() {
  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};

  // Initial invocation
  performFusion(q, range<1>{dataSize / 2});
  // CHECK: JIT DEBUG: Compiling new kernel, no suitable cached kernel found

  // Identical invocation but with a different range. This should lead to JIT
  // cache hit because the kernels were fused with homogeneous ranges, meaning
  // no size/ID remapping took place.
  // Regression: The output verification will fail if the kernel is launched
  // with the original range (when the fused kernel first entered the cache),
  // instead of the range supplied here.
  performFusion(q, range<1>{dataSize});
  // CHECK-NEXT: JIT DEBUG: Re-using cached JIT kernel
  // CHECK-NEXT: INFO: Re-using existing device binary for fused kernel

  return 0;
}
