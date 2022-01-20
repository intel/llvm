// Check that SYCLLowerWGLocalMemory pass is added to the SYCL device
// compilation pipeline with the inliner pass.

// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm \
// RUN:   -flegacy-pass-manager -mllvm -debug-pass=Structure %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=CHECK-INL,CHECK

// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm -O0 \
// RUN:   -flegacy-pass-manager -mllvm -debug-pass=Structure %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK-ALWINL,CHECK

// Check that AlwaysInliner pass is always run for compilation of SYCL device
// target code, even if all optimizations are disabled.

// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm -disable-llvm-passes \
// RUN:   -flegacy-pass-manager -mllvm -debug-pass=Structure %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK-ALWINL,CHECK
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm -fno-sycl-early-optimizations \
// RUN:   -flegacy-pass-manager -mllvm -debug-pass=Structure %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK-ALWINL,CHECK

// CHECK-INL: Function Integration/Inlining
// CHECK-ALWINL: Inliner for always_inline functions
// CHECK: Replace __sycl_allocateLocalMemory with allocation of memory in local address space
