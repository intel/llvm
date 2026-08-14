//==-- per-arch-backend-options.cpp ---------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// End-to-end test: build a SYCL program AOT-compiled for two Intel GPU archs
// (pvc + dg2_g10) with distinct -Xsycl-target-backend options for each, and
// verify that:
//   1. The driver emits per-(triple, arch) --device-compiler entries and
//      the wrapper routes each arch's tokens to its own ocloc invocation
//      (compile-time check on -v output).
//   2. The resulting fat binary runs correctly on a matching device
//      (runtime check).

// REQUIRES: ocloc, target-spir
// REQUIRES: arch-intel_gpu_pvc
// One physical arch (pvc) is required so %{run} has an AOT image matching
// the local device. The second arch (dg2_g10) is a build-only target that
// exercises the routing logic; ocloc builds its image but the runtime never
// executes it.

// RUN: %clangxx -Wno-error=unused-command-line-argument \
// RUN:   --offload-new-driver -fsycl \
// RUN:   -fsycl-targets=intel_gpu_pvc,intel_gpu_dg2_g10 \
// RUN:   -Xsycl-target-backend=intel_gpu_pvc "-options -cl-mad-enable" \
// RUN:   -Xsycl-target-backend=intel_gpu_dg2_g10 "-options -cl-unsafe-math-optimizations" \
// RUN:   -v %s -o %t.out > %t.log 2>&1
// RUN: FileCheck --input-file=%t.log --check-prefix=CHECK-PVC %s
// RUN: FileCheck --input-file=%t.log --check-prefix=CHECK-ACM %s
// RUN: %{run} %t.out

// pvc's ocloc call carries -cl-mad-enable, NOT -cl-unsafe-math-optimizations.
// CHECK-PVC: ocloc{{.*}} -device pvc {{.*}}-cl-mad-enable
// CHECK-PVC-NOT: ocloc{{.*}} -device pvc {{.*}}-cl-unsafe-math-optimizations

// dg2_g10's canonical ocloc device name is acm_g10.
// CHECK-ACM: ocloc{{.*}} -device acm_g10 {{.*}}-cl-unsafe-math-optimizations
// CHECK-ACM-NOT: ocloc{{.*}} -device acm_g10 {{.*}}-cl-mad-enable

// Regression: raw spir64_gen with an embedded "-device <arch>" in the
// backend option value must also route per-arch without leakage.
// RUN: %clangxx -Wno-error=unused-command-line-argument \
// RUN:   --offload-new-driver -fsycl \
// RUN:   -fsycl-targets=intel_gpu_dg2_g10,spir64_gen \
// RUN:   -Xsycl-target-backend=spir64_gen "-device pvc -options -cl-mad-enable" \
// RUN:   -Xsycl-target-backend=intel_gpu_dg2_g10 "-options -cl-unsafe-math-optimizations" \
// RUN:   -v %s -o %t_raw.out > %t_raw.log 2>&1
// RUN: FileCheck --input-file=%t_raw.log --check-prefix=CHECK-RAW-PVC %s
// RUN: FileCheck --input-file=%t_raw.log --check-prefix=CHECK-RAW-ACM %s

// CHECK-RAW-PVC: ocloc{{.*}} -device pvc {{.*}}-cl-mad-enable
// CHECK-RAW-PVC-NOT: ocloc{{.*}} -device pvc {{.*}}-cl-unsafe-math-optimizations
// CHECK-RAW-ACM: ocloc{{.*}} -device acm_g10 {{.*}}-cl-unsafe-math-optimizations
// CHECK-RAW-ACM-NOT: ocloc{{.*}} -device acm_g10 {{.*}}-cl-mad-enable

#include <sycl/detail/core.hpp>

#include <array>
#include <cstddef>
#include <iostream>

constexpr std::size_t N = 16;

class VecAdd;

int main() {
  std::array<int, N> a{}, b{}, c{};
  for (std::size_t i = 0; i < N; ++i) {
    a[i] = static_cast<int>(i);
    b[i] = static_cast<int>(2 * i);
  }

  {
    sycl::queue q;
    sycl::buffer<int, 1> bufA{a.data(), sycl::range<1>{N}};
    sycl::buffer<int, 1> bufB{b.data(), sycl::range<1>{N}};
    sycl::buffer<int, 1> bufC{c.data(), sycl::range<1>{N}};

    q.submit([&](sycl::handler &h) {
       sycl::accessor accA{bufA, h, sycl::read_only};
       sycl::accessor accB{bufB, h, sycl::read_only};
       sycl::accessor accC{bufC, h, sycl::write_only};
       h.parallel_for<VecAdd>(sycl::range<1>{N}, [=](sycl::id<1> i) {
         accC[i] = accA[i] + accB[i];
       });
     }).wait();
  }

  for (std::size_t i = 0; i < N; ++i) {
    int expected = static_cast<int>(i) + static_cast<int>(2 * i);
    if (c[i] != expected) {
      std::cerr << "FAIL at i=" << i << ": got " << c[i] << ", expected "
                << expected << "\n";
      return 1;
    }
  }

  std::cout << "PASS\n";
  return 0;
}
