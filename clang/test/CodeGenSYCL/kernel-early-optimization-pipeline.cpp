// Check LLVM optimization pipeline is run by default for SPIR-V compiled for
// SYCL device target, and can be disabled with -fno-sycl-early-optimizations.
// New pass manager doesn't print all passes tree, only module level.
//
// RUN: %clang_cc1 -O2 -fsycl-is-device -triple spir64-unknown-unknown %s -mdebug-pass Structure -emit-llvm -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-NEWPM-EARLYOPT
// CHECK-NEWPM-EARLYOPT: ConstantMergePass
// CHECK-NEWPM-EARLYOPT: SYCLMutatePrintfAddrspacePass
//
// RUN: %clang_cc1 -O2 -fsycl-is-device -triple spir64-unknown-unknown %s -mdebug-pass Structure -emit-llvm -fno-sycl-early-optimizations -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-NEWPM-NOEARLYOPT
// CHECK-NEWPM-NOEARLYOPT-NOT: ConstantMergePass
// CHECK-NEWPM-NOEARLYOPT: SYCLMutatePrintfAddrspacePass
