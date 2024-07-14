// RUN: %clangxx %s -fsycl -fsycl-device-only -S -emit-llvm -o - | FileCheck %s

// Check the 'ext_oneapi_private_alloca' aspect is listed in the list of used
// aspects.

#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/alloca.hpp>

class Kernel0;
class Kernel1;

// CHECK-LABEL: spir_kernel void @_ZTS7Kernel0
// CHECK-SAME:      !sycl_used_aspects ![[#USED_ASPECTS:]]

// CHECK-LABEL: spir_kernel void @_ZTS7Kernel1
// CHECK-SAME:      !sycl_used_aspects ![[#USED_ASPECTS:]]

// CHECK: ![[#USED_ASPECTS]] = !{![[#ASPECT:]]}
// CHECK: ![[#ASPECT]] = !{!"ext_oneapi_private_alloca", i32 63}

constexpr static sycl::specialization_id<int> size(10);

SYCL_EXTERNAL void foo(sycl::id<1> i, int *a,
                       sycl::decorated_private_ptr<int> tmp);

void test0(sycl::queue q, sycl::range<1> r, int *a, int s) {
  q.submit([&](sycl::handler &cgh) {
    cgh.set_specialization_constant<size>(s);
    cgh.parallel_for<Kernel0>(r, [=](sycl::id<1> i, sycl::kernel_handler kh) {
      foo(i, a,
          sycl::ext::oneapi::experimental::private_alloca<
              int, size, sycl::access::decorated::yes>(kh));
    });
  });
}

void test1(sycl::queue q, sycl::range<1> r, int *a, int s) {
  q.submit([&](sycl::handler &cgh) {
    cgh.set_specialization_constant<size>(s);
    cgh.parallel_for<Kernel1>(r, [=](sycl::id<1> i, sycl::kernel_handler kh) {
      foo(i, a,
          sycl::ext::oneapi::experimental::aligned_private_alloca<
              int, alignof(int) * 2, size, sycl::access::decorated::yes>(kh));
    });
  });
}
