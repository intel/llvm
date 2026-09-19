//==-- per-arch-backend-options.cpp ---------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Build-only test that -Xsycl-target-backend options for two AOT Intel GPU
// archs (pvc + dg2_g10) reach only their own arch's ocloc invocation.

// REQUIRES: ocloc

// RUN: %clangxx -Wno-error=unused-command-line-argument \
// RUN:   --offload-new-driver -fsycl \
// RUN:   -fsycl-targets=intel_gpu_pvc,intel_gpu_dg2_g10 \
// RUN:   -Xsycl-target-backend=intel_gpu_pvc "-options -cl-mad-enable" \
// RUN:   -Xsycl-target-backend=intel_gpu_dg2_g10 "-options -cl-unsafe-math-optimizations" \
// RUN:   -v %s -o %t.out > %t.log 2>&1
// RUN: FileCheck --input-file=%t.log --check-prefix=CHECK-PVC \
// RUN:   --implicit-check-not='ocloc{{.*}} -device pvc {{.*}}-cl-unsafe-math-optimizations' %s
// RUN: FileCheck --input-file=%t.log --check-prefix=CHECK-ACM \
// RUN:   --implicit-check-not='-device acm_g10 {{.*}}-cl-mad-enable' %s

// CHECK-PVC: ocloc{{.*}} -device pvc {{.*}}-cl-mad-enable
// dg2_g10's canonical ocloc device name is acm_g10.
// CHECK-ACM: ocloc{{.*}} -device acm_g10 {{.*}}-cl-unsafe-math-optimizations

#include <sycl/detail/core.hpp>

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>{1}, [=](sycl::id<1>) {});
  });
  return 0;
}
