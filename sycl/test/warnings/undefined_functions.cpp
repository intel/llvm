// RUN: %clangxx %s -fsycl -fsycl-link 2>&1 | FileCheck %s --check-prefix=CHECK-WARNING
// RUN: %clangxx %s -fsycl -fsycl-link -fsycl-allow-device-image-dependencies 2>&1 | FileCheck --allow-empty %s --check-prefix=CHECK-WARNING-DYNAMIC
// RUN: %clangxx %s -fsycl -fsycl-link -Wno-sycl-undefined-func-in-image 2>&1 | FileCheck --allow-empty %s --check-prefix=CHECK-WARNING-SUPPRESSED
// This test checks that sycl-post-link emits the "Undefined function ..."
// warning by default, that -fsycl-allow-device-image-dependencies suppresses
// it (existing behaviour), and that -Wno-sycl-undefined-func-in-image
// suppresses it (end-to-end coverage that the -W flag is threaded from the
// driver all the way through sycl-post-link and honoured in the output).
// Last-W-wins semantics is a pure driver-argument concern and is covered by
// clang/test/Driver/sycl-suppress-undefined-func-warnings.cpp; no need to
// re-run a full -fsycl-link compile here.

// CHECK-WARNING: warning: Undefined function _Z11external_f1ii found in
// CHECK-WARNING-DYNAMIC-NOT: warning: Undefined function _Z11external_f1ii found in
// CHECK-WARNING-SUPPRESSED-NOT: warning: Undefined function _Z11external_f1ii found in

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
