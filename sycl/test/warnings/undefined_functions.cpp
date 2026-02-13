// RUN: %clangxx %s -fsycl -fsycl-link 2>&1 | FileCheck %s --check-prefix=CHECK-WARNING
// RUN: %clangxx %s -fsycl -fsycl-link -fsycl-allow-device-image-dependencies 2>&1 | FileCheck --allow-empty %s --check-prefix=CHECK-WARNING-DYNAMIC
// This test is intended to check that we emit a helpful warning message for
// undefined user functions in a fully linked device image after the
// sycl-post-link stage of compilation.

// CHECK-WARNING: warning: Undefined function _Z11external_f1ii found in
// CHECK-WARNING-DYNAMIC-NOT: warning: Undefined function _Z11external_f1ii found in

#include <sycl/sycl.hpp>

SYCL_EXTERNAL int external_f1(int A, int B);

void hostf(unsigned Size, sycl::buffer<int, 1> &bufA,
           sycl::buffer<int, 1> &bufB, sycl::buffer<int, 1> &bufC) {
  sycl::range<1> range{Size};
  sycl::queue().submit([&](sycl::handler &cgh) {
    auto accA = bufA.get_access<sycl::access::mode::read>(cgh);
    auto accB = bufB.get_access<sycl::access::mode::read>(cgh);
    auto accC = bufC.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for<class Test>(range, [=](sycl::id<1> ID) {
      accC[ID] = external_f1(accA[ID], accB[ID]);
    });
  });
}
