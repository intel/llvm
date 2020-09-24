// UNSUPPORTED:  windows
// Test if clang is able to print internal defines in SYCL mode
// REQUIRES: x86-registered-target
//
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -dM -E -x c++ %s 2>&1 \
// RUN: | FileCheck --check-prefix CHECK-PRINT-INTERNAL-DEFINES %s
// CHECK-PRINT-INTERNAL-DEFINES: #define
