/// Check if -fsycl-instrument-device-code is allowed only for spir target

// RUN: %clang_cc1 -fsycl-instrument-device-code -triple spir-unknown-unknown %s -emit-llvm -o - 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsycl-instrument-device-code -triple spir64-unknown-unknown %s -emit-llvm -o - 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsycl-instrument-device-code -triple spir64_gen-unknown-unknown %s -emit-llvm -o - 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsycl-instrument-device-code -triple spir64_fpga-unknown-unknown %s -emit-llvm -o - 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsycl-instrument-device-code -triple spir64_x86_64-unknown-unknown %s -emit-llvm -o - 2>&1 | FileCheck %s
// CHECK-NOT: error

// RUN: not %clang_cc1 -fsycl-instrument-device-code -triple spirv32 -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERR
// RUN: not %clang_cc1 -fsycl-instrument-device-code -triple spirv64 -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERR
// RUN: not %clang_cc1 -fsycl-instrument-device-code -triple x86_64-unknown-unknown -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERR
// RUN: not %clang_cc1 -fsycl-instrument-device-code -triple i386-unknown-unknown -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERR
// RUN: not %clang_cc1 -fsycl-instrument-device-code -triple xcore-unknown-unknown -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERR
// RUN: not %clang_cc1 -fsycl-instrument-device-code -triple powerpc64-unknown-unknown -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERR
// RUN: not %clang_cc1 -fsycl-instrument-device-code -triple armv7-unknown-unknown -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERR
// CHECK-ERR: error: unsupported option '-fsycl-instrument-device-code' for target
