// Check that SYCLLowerWGLocalMemory pass is added to the SYCL device compilation pipeline with the inliner pass.

// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -emit-llvm \
// RUN:   -mllvm -debug-pass=Structure %s -o - 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-EARLYOPT
// CHECK-EARLYOPT: Function Integration/Inlining
// CHECK-EARLYOPT: Replace __sycl_allocateLocalMemory with allocation of memory in local address space

// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -emit-llvm \
// RUN:   -mllvm -debug-pass=Structure %s -o - -O0 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-O0opt
// CHECK-O0opt: Inliner for always_inline functions
// CHECK-O0opt: Replace __sycl_allocateLocalMemory with allocation of memory in local address space

// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -emit-llvm \
// RUN:   -mllvm -debug-pass=Structure %s -o - -disable-llvm-passes 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPASSES
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -emit-llvm \
// RUN:   -mllvm -debug-pass=Structure %s -o - -fno-sycl-early-optimizations 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPASSES
// CHECK-NOPASSES-NOT: Replace __sycl_allocateLocalMemory with allocation of memory in local address space
