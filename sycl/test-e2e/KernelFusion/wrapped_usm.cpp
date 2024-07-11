// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} %{embed-ir} -o %t.out
// RUN: env SYCL_PI_TRACE=2 %{run} %t.out | FileCheck %s

// Test complete fusion using an wrapped USM pointer as kernel functor argument.

// The two kernels are fused, so only a single, fused kernel is launched.
// CHECK-COUNT-1: piEnqueueKernelLaunch
// CHECK-NOT: piEnqueueKernelLaunch

#include <sycl/detail/core.hpp>
#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

template <typename T> struct wrapper {
  T *data;

  wrapper(size_t dataSize, queue &q)
      : data{sycl::malloc_shared<int>(dataSize, q)} {}

  T &operator[](size_t i) { return data[i]; }
  const T &operator[](size_t i) const { return data[i]; }
};

int main() {
  constexpr size_t dataSize = 512;

  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};

  wrapper<int> in1{dataSize, q};
  wrapper<int> in2{dataSize, q};
  wrapper<int> in3{dataSize, q};
  wrapper<int> tmp{dataSize, q};
  wrapper<int> out{dataSize, q};

  for (size_t i = 0; i < dataSize; ++i) {
    in1[i] = i * 2;
    in2[i] = i * 3;
    in3[i] = i * 4;
    tmp[i] = -1;
    out[i] = -1;
  }

  ext::codeplay::experimental::fusion_wrapper fw{q};
  fw.start_fusion();

  assert(fw.is_in_fusion_mode() && "Queue should be in fusion mode");

  q.submit([&](handler &cgh) {
    cgh.parallel_for<class KernelOne>(
        dataSize, [=](id<1> i) { tmp.data[i] = in1.data[i] + in2.data[i]; });
  });

  q.submit([&](handler &cgh) {
    cgh.parallel_for<class KernelTwo>(
        dataSize, [=](id<1> i) { out.data[i] = tmp.data[i] * in3.data[i]; });
  });

  fw.complete_fusion({ext::codeplay::experimental::property::no_barriers{}});

  assert(!fw.is_in_fusion_mode() &&
         "Queue should not be in fusion mode anymore");

  q.wait();

  // Check the results
  for (size_t i = 0; i < dataSize; ++i) {
    assert(out[i] == (20 * i * i) && "Computation error");
  }

  sycl::free(in1.data, q);
  sycl::free(in2.data, q);
  sycl::free(in3.data, q);
  sycl::free(tmp.data, q);
  sycl::free(out.data, q);

  return 0;
}
