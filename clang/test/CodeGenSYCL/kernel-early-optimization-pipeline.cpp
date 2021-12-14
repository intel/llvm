// Check LLVM optimization pipeline is run by default for SPIR-V compiled for
// SYCL device target, and can be disabled with -fno-sycl-early-optimizations.
//
// New pass manager doesn't print all passes tree, only module level.
// Also another option "-mdebug-pass Structure" should be used to print passes
// run by new PM. Right now this test is executed for legacy pass manager
// because pass structure output differs for new PM.
// TODO: rewrite test to perform checks under new pass manager.
//
// RUN: %clang_cc1 -O2 -fsycl-is-device -triple spir64-unknown-unknown %s -flegacy-pass-manager -mllvm -debug-pass=Structure -emit-llvm -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-EARLYOPT
// CHECK-EARLYOPT: Lower Work Group Scope Code
// CHECK-EARLYOPT: Combine redundant instructions
//
// RUN: %clang_cc1 -O2 -fsycl-is-device -fno-sycl-early-optimizations -triple spir64-unknown-unknown %s -flegacy-pass-manager -mllvm -debug-pass=Structure -emit-llvm -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-NOEARLYOPT
// CHECK-NOEARLYOPT: Lower Work Group Scope Code
// CHECK-NOEARLYOPT-NOT: Combine redundant instructions
