// Check LLVM optimization pipeline is run by default for SPIR-V compiled for
// SYCL device target, and can be disabled with -fno-sycl-early-optimizations.
//
// RUN: %clang_cc1 -O2 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice %s -mllvm -debug-pass=Structure -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-EARLYOPT
// CHECK-EARLYOPT: Lower Work Group Scope Code
// CHECK-EARLYOPT: Combine redundant instructions
//
// RUN: %clang_cc1 -O2 -fsycl-is-device -fno-sycl-early-optimizations -triple spir64-unknown-unknown-sycldevice %s -mllvm -debug-pass=Structure -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-NOEARLYOPT
// CHECK-NOEARLYOPT: Lower Work Group Scope Code
// CHECK-NOEARLYOPT-NOT: Combine redundant instructions
