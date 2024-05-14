// RUN: %{build} -fsycl-embed-ir -o %t.out
// RUN: env SYCL_RT_WARNING_LEVEL=1 %{run} %t.out 2>&1 | FileCheck %s

// Windows doesn't yet have full shutdown().
// UNSUPPORTED: ze_debug && windows

// Test fusion cancellation on host task submission happening before
// complete_fusion.

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
      auto accIn1 = bIn1.get_access<access::mode::read>(cgh);
      auto accIn2 = bIn2.get_access<access::mode::read>(cgh);
      auto accTmp = bTmp.get_access(cgh);
      cgh.parallel_for<class KernelOne>(
          dataSize, [=](id<1> i) { accTmp[i] = accIn1[i] + accIn2[i]; });
    });

    q.submit([&](handler &cgh) {
      auto accTmp = bTmp.get_access(cgh);
      auto accIn3 = bIn3.get_access(cgh);
      auto accOut = bOut.get_access(cgh);
      cgh.parallel_for<class KernelTwo>(
          dataSize, [=](id<1> i) { accOut[i] = accTmp[i] * accIn3[i]; });
    });

    // This host task requests access to bOut, which is accessed by one of
    // the kernels in the fusion list, creating a requirement for one of the
    // kernels in the fusion list. This should lead to cancellation of the
    // fusion.
    q.submit([&](handler &cgh) {
      auto accOut = bOut.get_access(cgh);
      cgh.host_task([=]() { accOut[256] = 42; });
    });

    assert(!fw.is_in_fusion_mode() &&
           "Queue should not be in fusion mode anymore");

    fw.complete_fusion({ext::codeplay::experimental::property::no_barriers{}});
  }

  // Check the results
  for (size_t i = 0; i < dataSize; ++i) {
    if (i == 256) {
      assert(out[i] == 42 && "Computation error");
    } else {
      assert(out[i] == (20 * i * i) && "Computation error");
    }
  }

  return 0;
}

// CHECK: WARNING: Aborting fusion because synchronization with one of the kernels in the fusion list was requested
