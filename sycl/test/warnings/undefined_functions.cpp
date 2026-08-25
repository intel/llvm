// RUN: %clangxx %s -fsycl -fsycl-link 2>&1 | FileCheck %s --check-prefix=CHECK-WARNING
// RUN: %clangxx %s -fsycl -fsycl-link -fsycl-allow-device-image-dependencies 2>&1 | FileCheck --allow-empty %s --check-prefix=CHECK-WARNING-DYNAMIC
// RUN: %clangxx %s -fsycl -fsycl-link -Wno-sycl-undefined-func-in-image 2>&1 | FileCheck --allow-empty %s --check-prefix=CHECK-WARNING-SUPPRESSED
// RUN: %clangxx %s -fsycl -fsycl-link -Wno-sycl-undefined-func-in-image -Wsycl-undefined-func-in-image 2>&1 | FileCheck %s --check-prefix=CHECK-WARNING-REENABLED
// RUN: %clangxx %s -fsycl -fsycl-link -Wsycl-undefined-func-in-image -Wno-sycl-undefined-func-in-image 2>&1 | FileCheck --allow-empty %s --check-prefix=CHECK-WARNING-LAST-WNO
// This test is intended to check that we emit a helpful warning message for
// undefined user functions in a fully linked device image after the
// sycl-post-link stage of compilation.
//
// -Wno-sycl-undefined-func-in-image suppresses the warning by forwarding
// -suppress-undefined-func-warnings to sycl-post-link. A later
// -Wsycl-undefined-func-in-image on the same command line re-enables it,
// and a later -Wno- suppresses it again -- last -W... wins in either
// direction, matching normal -W option semantics.

// CHECK-WARNING: warning: Undefined function _Z11external_f1ii found in
// CHECK-WARNING-DYNAMIC-NOT: warning: Undefined function _Z11external_f1ii found in
// CHECK-WARNING-SUPPRESSED-NOT: warning: Undefined function _Z11external_f1ii found in
// CHECK-WARNING-REENABLED: warning: Undefined function _Z11external_f1ii found in
// CHECK-WARNING-LAST-WNO-NOT: warning: Undefined function _Z11external_f1ii found in

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
