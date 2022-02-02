// Check LLVM optimization pipeline is run by default for SPIR-V compiled for
// SYCL device target, and can be disabled with -fno-sycl-early-optimizations.
//
// RUN: %clang_cc1 -O2 -fsycl-is-device -triple spir64-unknown-unknown %s -flegacy-pass-manager -mllvm -debug-pass=Structure -emit-llvm -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-EARLYOPT
// CHECK-EARLYOPT: Combine redundant instructions
// CHECK-EARLYOPT: Move SYCL printf literal arguments to constant address space
//
// RUN: %clang_cc1 -O2 -fsycl-is-device -triple spir64-unknown-unknown %s -flegacy-pass-manager -mllvm -debug-pass=Structure -emit-llvm -fno-sycl-early-optimizations -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-NOEARLYOPT
// CHECK-NOEARLYOPT-NOT: Combine redundant instructions
// CHECK-NOEARLYOPT: Move SYCL printf literal arguments to constant address space
//
//
// New pass manager doesn't print all passes tree, only module level.
//
// RUN: %clang_cc1 -O2 -fsycl-is-device -triple spir64-unknown-unknown %s -fno-legacy-pass-manager -mdebug-pass Structure -emit-llvm -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-NEWPM-EARLYOPT
// CHECK-NEWPM-EARLYOPT: ConstantMergePass
// CHECK-NEWPM-EARLYOPT: SYCLMutatePrintfAddrspacePass
//
// RUN: %clang_cc1 -O2 -fsycl-is-device -triple spir64-unknown-unknown %s -fno-legacy-pass-manager -mdebug-pass Structure -emit-llvm -fno-sycl-early-optimizations -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-NEWPM-NOEARLYOPT
// CHECK-NEWPM-NOEARLYOPT-NOT: ConstantMergePass
// CHECK-NEWPM-NOEARLYOPT: SYCLMutatePrintfAddrspacePass
