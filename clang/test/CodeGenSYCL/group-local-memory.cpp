// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice \
// RUN:     -S -emit-llvm -mllvm -debug-pass=Structure -disable-llvm-passes \
// RUN:     -o - %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice \
// RUN:     -S -emit-llvm -mllvm -debug-pass=Structure -o - %s 2>&1 \
// RUN:     | FileCheck %s

// CHECK: Replace __sycl_allocateLocalMemory with allocation of memory in local address space
