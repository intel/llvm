// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fsycl-int-header=%t.h %s
// RUN: FileCheck -input-file=%t.h %s
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fsycl-range-rounding=disable -fsycl-int-header=%t.h %s
// RUN: FileCheck -input-file=%t.h %s --check-prefix=CHECK-RANGE
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fsycl-range-rounding=force -fsycl-int-header=%t.h %s
// RUN: FileCheck -input-file=%t.h %s --check-prefix=CHECK-FORCE-RANGE
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fsycl-int-header=%t.h %s
// RUN: FileCheck -input-file=%t.h %s --check-prefix=CHECK-NO-RANGE

// Test verifying predefines which need to be set for host and device compilation.
// Preprocessor macros which are required when using custom host compiler must be
// defined in integration header.

#include "Inputs/sycl.hpp"

int main() {
  sycl::kernel_single_task<class first_kernel>([]() {});
}
// CHECK: #ifndef SYCL_LANGUAGE_VERSION
// CHECK-NEXT: #define SYCL_LANGUAGE_VERSION 202001
// CHECK-NEXT: #endif //SYCL_LANGUAGE_VERSION

// CHECK-RANGE: #ifndef __SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__
// CHECK-RANGE-NEXT: #define __SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__ 1
// CHECK-RANGE-NEXT: #endif //__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__

// CHECK-FORCE-RANGE: #ifndef __SYCL_FORCE_PARALLEL_FOR_RANGE_ROUNDING__
// CHECK-FORCE-RANGE-NEXT: #define __SYCL_FORCE_PARALLEL_FOR_RANGE_ROUNDING__ 1
// CHECK-FORCE-RANGE-NEXT: #endif //__SYCL_FORCE_PARALLEL_FOR_RANGE_ROUNDING__

// CHECK-NO-RANGE-NOT: #define __SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__ 1
// CHECK-NO-RANGE-NOT: #define __SYCL_FORCE_PARALLEL_FOR_RANGE_ROUNDING__ 1
