// Check that SYCLLowerWGLocalMemory pass is added to the SYCL device
// compilation pipeline with the inliner pass (new Pass Manager).

// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm \
// RUN:   -fno-legacy-pass-manager -mdebug-pass Structure %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=CHECK-INL,CHECK

// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm -O0 \
// RUN:   -fno-legacy-pass-manager -mdebug-pass Structure %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK-ALWINL,CHECK

// Check that AlwaysInliner pass is always run for compilation of SYCL device
// target code, even if all optimizations are disabled.

// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm -disable-llvm-passes \
// RUN:   -fno-legacy-pass-manager -mdebug-pass Structure %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK-ALWINL,CHECK
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm -fno-sycl-early-optimizations \
// RUN:   -fno-legacy-pass-manager -mdebug-pass Structure %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK-ALWINL,CHECK

// CHECK-INL: Running pass: ModuleInlinerWrapperPass on [module]
// CHECK-ALWINL: Running pass: AlwaysInlinerPass on [module]
// CHECK: Running pass: SYCLLowerWGLocalMemoryPass on [module]
