// RUN: %{build} %{embed-ir} -O2 -o %t.out
// RUN: env SYCL_RT_WARNING_LEVEL=1 %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not "COMPUTATION ERROR" --implicit-check-not "WRONG INTERNALIZATION"

// Test caching for JIT fused kernels. Also test for debug messages being
// printed when SYCL_RT_WARNING_LEVEL=1.

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>
#include <sycl/properties/all_properties.hpp>

using namespace sycl;

constexpr size_t dataSize = 512;

enum class Internalization { None, Local, Private };

void performFusion(queue &q, Internalization internalize, range<1> globalSize,
                   int beta, int gamma, bool insertBarriers = false) {
  int alpha = 1;
  int in1[dataSize], in2[dataSize], in3[dataSize], tmp[dataSize], out[dataSize];

  for (size_t i = 0; i < dataSize; ++i) {
    in1[i] = i * 2;
    in2[i] = i * 3;
    in3[i] = i * 4;
    tmp[i] = -1;
    out[i] = -1;
  }
  {
    buffer<int> bIn1{in1, globalSize};
    buffer<int> bIn2{in2, globalSize};
    buffer<int> bIn3{in3, globalSize};
    buffer<int> bTmp{tmp, globalSize};
    buffer<int> bOut{out, globalSize};

    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();

    assert(fw.is_in_fusion_mode() && "Queue should be in fusion mode");

    q.submit([&](handler &cgh) {
      auto accIn1 = bIn1.get_access(cgh);
      auto accIn2 = bIn2.get_access(cgh);
      property_list properties{};
      if (internalize == Internalization::Private) {
        properties = {
            sycl::ext::codeplay::experimental::property::promote_private{}};
      } else if (internalize == Internalization::Local) {
        properties = {
            sycl::ext::codeplay::experimental::property::promote_local{}};
      }
      accessor<int> accTmp = bTmp.get_access(cgh, properties);
      cgh.parallel_for<class KernelOne>(globalSize, [=](id<1> i) {
        accTmp[i] = accIn1[i] + accIn2[i] * alpha;
      });
    });

    q.submit([&](handler &cgh) {
      property_list properties{};
      if (internalize == Internalization::Private) {
        properties = {
            sycl::ext::codeplay::experimental::property::promote_private{}};
      } else if (internalize == Internalization::Local) {
        properties = {
            sycl::ext::codeplay::experimental::property::promote_local{}};
      }
      accessor<int> accTmp = bTmp.get_access(cgh, properties);
      auto accIn3 = bIn3.get_access(cgh);
      auto accOut = bOut.get_access(cgh);
      cgh.parallel_for<class KernelTwo>(globalSize, [=](id<1> i) {
        accOut[i] = accTmp[i] * accIn3[i] * beta * gamma;
      });
    });

    if (insertBarriers) {
      fw.complete_fusion();
    } else {
      fw.complete_fusion(
          {ext::codeplay::experimental::property::no_barriers{}});
    }

    assert(!fw.is_in_fusion_mode() &&
           "Queue should not be in fusion mode anymore");
  }

  // Check the results
  size_t numErrors = 0;
  size_t numInternalized = 0;
  for (size_t i = 0; i < dataSize; ++i) {
    if (i < globalSize.size() && out[i] != (20 * i * i * beta * gamma)) {
      ++numErrors;
    }
    if (tmp[i] == -1) {
      ++numInternalized;
    }
  }
  if (numErrors) {
    std::cout << "COMPUTATION ERROR\n";
  }
  if ((internalize == Internalization::None) && numInternalized) {
    std::cout << "WRONG INTERNALIZATION\n";
  }
}

int main() {
  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};

  // Initial invocation
  performFusion(q, Internalization::Private, range<1>{dataSize}, 1, 1);
  // CHECK: JIT DEBUG: Compiling new kernel, no suitable cached kernel found

  // Identical invocation, should lead to JIT cache hit.
  performFusion(q, Internalization::Private, range<1>{dataSize}, 1, 1);
  // CHECK-NEXT: JIT DEBUG: Re-using cached JIT kernel
  // CHECK-NEXT: INFO: Re-using existing device binary for fused kernel

  // Invocation with a different beta. Because beta was identical to alpha so
  // far, this should lead to a cache miss.
  performFusion(q, Internalization::Private, range<1>{dataSize}, 2, 1);
  // CHECK-NEXT: JIT DEBUG: Compiling new kernel, no suitable cached kernel found

  // Invocation with barrier insertion should lead to a cache miss.
  performFusion(q, Internalization::Private, range<1>{dataSize}, 1, 1,
                /* insertBarriers */ true);
  // CHECK-NEXT: JIT DEBUG: Compiling new kernel, no suitable cached kernel found

  // Invocation with different internalization target should lead to a cache
  // miss.
  performFusion(q, Internalization::None, range<1>{dataSize}, 1, 1);
  // CHECK-NEXT: JIT DEBUG: Compiling new kernel, no suitable cached kernel found

  // Invocation with a different gamma should lead to a cache miss because gamma
  // participates in constant propagation.
  performFusion(q, Internalization::Private, range<1>{dataSize}, 1, 2);
  // CHECK-NEXT: JIT DEBUG: Compiling new kernel, no suitable cached kernel found

  return 0;
}
