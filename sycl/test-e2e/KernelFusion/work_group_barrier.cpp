// RUN: %{build} %{embed-ir} -o %t.out
// RUN: env SYCL_PI_TRACE=2 %{run} %t.out | FileCheck %s

// Test complete fusion with a combination of kernels that require a work-group
// barrier to be inserted by fusion.

// The two kernels are fused, so only a single, fused kernel is launched.
// CHECK-COUNT-1: piEnqueueKernelLaunch
// CHECK-NOT: piEnqueueKernelLaunch

#include <sycl/detail/core.hpp>
#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>
#include <sycl/properties/all_properties.hpp>

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

  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};

  {
    buffer<int> bIn1{in1, range{dataSize}};
    buffer<int> bIn2{in2, range{dataSize}};
    buffer<int> bIn3{in3, range{dataSize}};
    buffer<int> bTmp{tmp, range{dataSize}};
    buffer<int> bOut{out, range{dataSize}};

    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();

    assert(fw.is_in_fusion_mode() && "Queue should be in fusion mode");

    q.submit([&](handler &cgh) {
      auto accIn1 = bIn1.get_access(cgh);
      auto accIn2 = bIn2.get_access(cgh);
      auto accTmp = bTmp.get_access(cgh);
      cgh.parallel_for<class KernelOne>(
          nd_range<1>{{dataSize}, {32}}, [=](nd_item<1> i) {
            auto workgroupSize = i.get_local_range(0);
            auto baseOffset = i.get_group_linear_id() * workgroupSize;
            auto localIndex = i.get_local_linear_id();
            auto localOffset = (workgroupSize - 1) - localIndex;
            accTmp[baseOffset + localOffset] =
                accIn1[baseOffset + localOffset] +
                accIn2[baseOffset + localOffset];
          });
    });

    q.submit([&](handler &cgh) {
      auto accTmp = bTmp.get_access(cgh);
      auto accIn3 = bIn3.get_access(cgh);
      auto accOut = bOut.get_access(cgh);
      cgh.parallel_for<class KernelTwo>(
          nd_range<1>{{dataSize}, {32}}, [=](nd_item<1> i) {
            auto index = i.get_global_linear_id();
            accOut[index] = accTmp[index] * accIn3[index];
          });
    });

    fw.complete_fusion();

    assert(!fw.is_in_fusion_mode() &&
           "Queue should not be in fusion mode anymore");
  }

  // Check the results
  for (size_t i = 0; i < dataSize; ++i) {
    assert(out[i] == (20 * i * i) && "Computation error");
  }

  return 0;
}
