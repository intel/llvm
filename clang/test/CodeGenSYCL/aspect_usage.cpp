// Checks that Propagate aspect usage Pass is run for SYCL device target.
//
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown %s -flegacy-pass-manager -mllvm -debug-pass=Structure -emit-llvm -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-SYCL1
// CHECK-SYCL1: Propagate aspect usage
//
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown %s -flegacy-pass-manager -fno-sycl-early-optimizations -mllvm -debug-pass=Structure -emit-llvm -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-SYCL2
// CHECK-SYCL2: Propagate aspect usage
//
// RUN: %clang_cc1 %s -flegacy-pass-manager -mllvm -debug-pass=Structure -emit-llvm -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-NOT-SYCL1
// CHECK-NOT-SYCL1-NOT: Propagate aspect usage
//
//
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown %s -fno-legacy-pass-manager -mdebug-pass Structure -emit-llvm -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-NEWPM-SYCL1
// CHECK-NEWPM-SYCL1: PropagateAspectUsagePass
//
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown %s -fno-legacy-pass-manager -fno-sycl-early-optimizations -mdebug-pass Structure -emit-llvm -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-NEWPM-SYCL2
// CHECK-NEWPM-SYCL2: PropagateAspectUsagePass
//
// RUN: %clang_cc1 %s -fno-legacy-pass-manager -mdebug-pass Structure -emit-llvm -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-NEWPM-NOT-SYCL
// CHECK-NEWPM-NOT-SYCL-NOT: PropagateAspectUsagePass
