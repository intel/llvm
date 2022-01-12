// Check LLVM optimization pipeline is run by default for SPIR-V compiled for
// SYCL device target, and can be disabled with -fno-sycl-early-optimizations.
//
// RUN: %clang_cc1 -O2 -fsycl-is-device -triple spir64-unknown-unknown %s -flegacy-pass-manager -mllvm -debug-pass=Structure -emit-llvm -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-EARLYOPT
// CHECK-EARLYOPT: Lower Work Group Scope Code
// CHECK-EARLYOPT: Combine redundant instructions
//
// RUN: %clang_cc1 -O2 -fsycl-is-device -triple spir64-unknown-unknown %s -flegacy-pass-manager -mllvm -debug-pass=Structure -emit-llvm -fno-sycl-early-optimizations -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-NOEARLYOPT
// CHECK-NOEARLYOPT: Lower Work Group Scope Code
// CHECK-NOEARLYOPT-NOT: Combine redundant instructions
//
//
// New pass manager doesn't print all passes tree, only module level.
//
// RUN: %clang_cc1 -O2 -fsycl-is-device -triple spir64-unknown-unknown %s -fno-legacy-pass-manager -mdebug-pass Structure -emit-llvm -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-NEWPM-EARLYOPT
// CHECK-NEWPM-EARLYOPT: SYCLMutatePrintfAddrspacePass
// CHECK-NEWPM-EARLYOPT: ConstantMergePass
//
// RUN: %clang_cc1 -O2 -fsycl-is-device -triple spir64-unknown-unknown %s -fno-legacy-pass-manager -mdebug-pass Structure -emit-llvm -fno-sycl-early-optimizations -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-NEWPM-NOEARLYOPT
// CHECK-NEWPM-NOEARLYOPT: SYCLMutatePrintfAddrspacePass
// CHECK-NEWPM-NOEARLYOPT-NOT: ConstantMergePass
