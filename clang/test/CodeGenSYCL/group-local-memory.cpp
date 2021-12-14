// Check that SYCLLowerWGLocalMemory pass is added to the SYCL device
// compilation pipeline with the inliner pass.

// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm \
// RUN:   -mdebug-pass Structure %s -o - 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=CHECK-INL,CHECK

// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm \
// RUN:   -mdebug-pass Structure %s -o - -O0 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK-ALWINL,CHECK

// Check that AlwaysInliner pass is always run for compilation of SYCL device
// target code, even if all optimizations are disabled.

// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm \
// RUN:   -mdebug-pass Structure %s -o - -disable-llvm-passes 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=CHECK-ALWINL,CHECK
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm \
// RUN:   -mdebug-pass Structure %s -o - -fno-sycl-early-optimizations 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK-ALWINL,CHECK

// CHECK-INL: Running pass: ModuleInlinerWrapperPass on [module]
// CHECK-ALWINL: Running pass: AlwaysInlinerPass on [module]
// CHECK: Running pass: SYCLLowerWGLocalMemoryPass on [module]
