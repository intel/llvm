// RUN: %{build} %{embed-ir} -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s

// Test complete fusion with local internalization specified on the
// accessors, where each work-item processes multiple data-items.

// The two kernels are fused, so only a single, fused kernel is launched.
// CHECK-COUNT-1: urEnqueueKernelLaunch
// CHECK-NOT: urEnqueueKernelLaunch

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
      auto accTmp = bTmp.get_access(
          cgh, sycl::ext::codeplay::experimental::property::promote_local{});
      cgh.parallel_for<class KernelOne>(
          nd_range<1>{{128}, {8}}, [=](nd_item<1> ndi) {
            auto baseOffset = ndi.get_global_linear_id() * 4;
            for (size_t j = 0; j < 4; ++j) {
              accTmp[baseOffset + j] =
                  accIn1[baseOffset + j] + accIn2[baseOffset + j];
            }
          });
    });

    q.submit([&](handler &cgh) {
      auto accTmp = bTmp.get_access(
          cgh, sycl::ext::codeplay::experimental::property::promote_local{});
      auto accIn3 = bIn3.get_access(cgh);
      auto accOut = bOut.get_access(cgh);
      cgh.parallel_for<class KernelTwo>(
          nd_range<1>{{128}, {8}}, [=](nd_item<1> ndi) {
            auto baseOffset = ndi.get_global_linear_id() * 4;
            for (size_t j = 0; j < 4; ++j) {
              accOut[baseOffset + j] =
                  accTmp[baseOffset + j] * accIn3[baseOffset + j];
            }
          });
    });

    fw.complete_fusion({ext::codeplay::experimental::property::no_barriers{}});

    assert(!fw.is_in_fusion_mode() &&
           "Queue should not be in fusion mode anymore");
  }

  // Check the results
  for (size_t i = 0; i < dataSize; ++i) {
    assert(out[i] == (20 * i * i) && "Computation error");
  }

  return 0;
}
