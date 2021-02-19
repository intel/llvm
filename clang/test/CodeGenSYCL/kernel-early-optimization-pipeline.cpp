// Check LLVM optimization pipeline is run by default for SPIR-V compiled for
// SYCL device target, and can be disabled with -fno-sycl-early-optimizations.
//
// RUN: %clang_cc1 -O2 -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice %s -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-EARLYOPT
// CHECK-EARLYOPT: Starting llvm::Module pass manager run.
// CHECK-EARLYOPT: Running pass: GlobalOptPass
// CHECK-EARLYOPT: Running pass: GlobalDCEPass
// CHECK-EARLYOPT: Running pass: PrintModulePass on
//
// RUN: %clang_cc1 -O2 -fsycl -fsycl-is-device -fno-sycl-early-optimizations -triple spir64-unknown-unknown-sycldevice %s -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-NOEARLYOPT
// CHECK-NOEARLYOPT: Starting llvm::Module pass manager run.
// CHECK-NOEARLYOPT-NEXT: Running pass: PrintModulePass on
