// RUN: %clangxx %s -fsycl -fsycl-device-only -S -emit-llvm -o - | FileCheck %s

// Check the 'ext_oneapi_private_alloca' aspect is listed in the list of used
// aspects.

#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/alloca.hpp>

class Kernel;

// CHECK-LABEL: spir_kernel void @_ZTS6Kernel
// CHECK-SAME:      !sycl_used_aspects ![[#USED_ASPECTS:]]

// CHECK:       ![[#USED_ASPECTS]] = !{i32 64}

constexpr static sycl::specialization_id<int> size(10);

SYCL_EXTERNAL void foo(sycl::id<1> i, int *a,
                       sycl::decorated_private_ptr<int> tmp);

void test(sycl::queue q, sycl::range<1> r, int *a, int s) {
  q.submit([&](sycl::handler &cgh) {
    cgh.set_specialization_constant<size>(s);
    cgh.parallel_for<Kernel>(r, [=](sycl::id<1> i, sycl::kernel_handler kh) {
      foo(i, a,
          sycl::ext::oneapi::experimental::private_alloca<
              int, size, sycl::access::decorated::yes>(kh));
    });
  });
}
