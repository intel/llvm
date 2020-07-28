// Test if sycl is able to print internal defines
//
// RUN: %clangxx -fsycl -dM -E -x c++ %s 2>&1 \
// RUN: | FileCheck --check-prefix CHECK-PRINT-INTERNAL-DEFINES %s
// CHECK-PRINT-INTERNAL-DEFINES: #define
