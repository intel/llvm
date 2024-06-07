// RUN: %{build} -fsycl-embed-ir -o %t.out
// RUN: env SYCL_RT_WARNING_LEVEL=1 %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not "Computation error" --implicit-check-not "Internalized" --check-prefix=CHECK %if hip %{ --check-prefix=CHECK-HIP %} %else %{ --check-prefix=CHECK-NON-HIP %}

// Test pointers being stored are not internalized.

// CHECK:      Unable to perform all promotions for function {{.*}}. Detailed information:
// CHECK-NON-HIP-NEXT: Failed to promote argument 0 of function {{.*}}: It is not safe to promote values being stored to another pointer
// COM: The libspirv for HIP adds an instruction prior to the store causing the
// internalization failure. COM: The failure is still related to what we expect,
// it just fails for a slightly different reason.
// CHECK-HIP-NEXT: Failed to promote argument 0 of function {{.*}}: Do not know how to handle value to promote

#include <sycl/detail/core.hpp>
#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>
#include <sycl/properties/all_properties.hpp>

#include <array>

using namespace sycl;

// Pointer will be stored in an accessor struct before being passed here.
__attribute__((noinline)) void
kernel_one_impl(accessor<int, 1, access::mode::read_write> acc, std::size_t i,
                int lhs, int rhs) {
  acc[i] = lhs + rhs;
}

int main() {
  constexpr size_t dataSize = 512;

  std::array<int, dataSize> in1, in2, in3, tmp, out;

  for (size_t i = 0; i < dataSize; ++i) {
    in1[i] = i * 2;
    in2[i] = i * 3;
    in3[i] = i * 4;
    tmp[i] = -1;
    out[i] = -1;
  }

  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};

  {
    buffer<int> bIn1{in1};
    buffer<int> bIn2{in2};
    buffer<int> bIn3{in3};
    buffer<int> bTmp{tmp};
    buffer<int> bOut{out};

    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();

    assert(fw.is_in_fusion_mode() && "Queue should be in fusion mode");

    q.submit([&](handler &cgh) {
      auto accIn1 = bIn1.get_access(cgh);
      auto accIn2 = bIn2.get_access(cgh);
      auto accTmp = bTmp.get_access(
          cgh, sycl::ext::codeplay::experimental::property::promote_private{});
      cgh.parallel_for<class KernelOne>(dataSize, [=](id<1> i) {
        kernel_one_impl(accTmp, i, accIn1[i], accIn2[i]);
      });
    });

    q.submit([&](handler &cgh) {
      auto accTmp = bTmp.get_access(
          cgh, sycl::ext::codeplay::experimental::property::promote_private{});
      auto accIn3 = bIn3.get_access(cgh);
      auto accOut = bOut.get_access(cgh);
      cgh.parallel_for<class KernelTwo>(
          dataSize, [=](id<1> i) { accOut[i] = accTmp[i] * accIn3[i]; });
    });

    fw.complete_fusion({ext::codeplay::experimental::property::no_barriers{}});

    assert(!fw.is_in_fusion_mode() &&
           "Queue should not be in fusion mode anymore");
  }

  // Check the results
  for (size_t i = 0; i < dataSize; ++i) {
    assert(out[i] == (20 * i * i) && "Computation error");
    assert(tmp[i] == (5 * i) && "Internalized");
  }
}
