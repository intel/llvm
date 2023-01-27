// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_RT_WARNING_LEVEL=1 %CPU_RUN_PLACEHOLDER %t.out 2>&1\
// RUN: %CPU_CHECK_PLACEHOLDER
// RUN: env SYCL_RT_WARNING_LEVEL=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1\
// RUN: %GPU_CHECK_PLACEHOLDER
// UNSUPPORTED: cuda || hip
// For this test, complete_fusion must be supported.
// REQUIRES: fusion

// Test fusion cancellation for requirement between two active fusions.

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  constexpr size_t dataSize = 512;
  int in1[dataSize], in2[dataSize], in3[dataSize], tmp[dataSize], out[dataSize];

  for (size_t i = 0; i < dataSize; ++i) {
    in1[i] = i * 2;
    in2[i] = i * 3;
    in3[i] = i * 4;
    tmp[i] = -1;
    out[i] = -1;
  }

  queue q1{ext::codeplay::experimental::property::queue::enable_fusion{}};
  queue q2{ext::codeplay::experimental::property::queue::enable_fusion{}};

  {
    buffer<int> bIn1{in1, range{dataSize}};
    buffer<int> bIn2{in2, range{dataSize}};
    buffer<int> bTmp{tmp, range{dataSize}};
    buffer<int> bOut{out, range{dataSize}};
    buffer<int> bIn3{in3, range{dataSize}};

    ext::codeplay::experimental::fusion_wrapper fw1{q1};
    fw1.start_fusion();

    assert(fw1.is_in_fusion_mode() && "Queue should be in fusion mode");

    q1.submit([&](handler &cgh) {
      auto accIn1 = bIn1.get_access<access::mode::read>(cgh);
      auto accIn2 = bIn2.get_access<access::mode::read>(cgh);
      auto accTmp = bTmp.get_access(cgh);
      cgh.parallel_for<class KernelOne>(
          dataSize, [=](id<1> i) { accTmp[i] = accIn1[i] + accIn2[i]; });
    });

    ext::codeplay::experimental::fusion_wrapper fw2{q2};
    fw2.start_fusion();

    q2.submit([&](handler &cgh) {
      auto accTmp = bTmp.get_access(cgh);
      cgh.parallel_for<class KernelThree>(dataSize,
                                          [=](id<1> i) { accTmp[i] *= 2; });
    });

    // KernelThree specifies a requirement on KernelOne. To avoid circular
    // dependencies between two fusions, the fusion for q1 needs to cancelled.
    assert(!fw1.is_in_fusion_mode() &&
           "Queue should not be in fusion mode anymore");

    assert(fw2.is_in_fusion_mode() && "Queue should be in fusion mode");

    q1.submit([&](handler &cgh) {
      auto accTmp = bTmp.get_access(cgh);
      auto accIn3 = bIn3.get_access(cgh);
      auto accOut = bOut.get_access(cgh);
      cgh.parallel_for<class KernelTwo>(
          dataSize, [=](id<1> i) { accOut[i] = accTmp[i] * accIn3[i]; });
    });

    // KernelTwo specifies a requirement on KernelThree, which leads to
    // cancellation of the fusion for q2.
    assert(!fw2.is_in_fusion_mode() &&
           "Queue should not be in fusion mode anymore");

    fw1.complete_fusion({ext::codeplay::experimental::property::no_barriers{}});

    fw2.cancel_fusion();
  }

  // Check the results
  for (size_t i = 0; i < dataSize; ++i) {
    assert(out[i] == (40 * i * i) && "Computation error");
  }

  return 0;
}

// CHECK: WARNING: Aborting fusion because of requirement from a different fusion
// CHECK-NEXT: WARNING: Aborting fusion because synchronization with one of the kernels in the fusion list was requested
