// RUN: %clang_cc1 %s -fsycl-is-device -triple nvptx64-nvidia-nvcl -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-NVPTX %s
// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64-unknown-unknown -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-NVPTX-NEG %s
// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64_gen-unknown-unknown -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-NVPTX-NEG %s
// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64_x86_64-unknown-unknown -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-NVPTX-NEG %s
// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64_fpga-unknown-unknown -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-NVPTX-NEG %s
// CHECK-NVPTX: #define __NVPTX__
// CHECK-NVPTX-NEG-NOT: #define __NVPTX__

// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64-unknown-unknown -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-SYCL-FP-ATOMICS %s
// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64_gen-unknown-unknown -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-SYCL-FP-ATOMICS %s
// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64_x86_64-unknown-unknown -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-SYCL-FP-ATOMICS %s
// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64_fpga-unknown-unknown -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-SYCL-FP-ATOMICS-NEG %s
// RUN: %clang_cc1 %s -fsycl-is-device -triple nvptx64-nvidia-nvcl -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-SYCL-FP-ATOMICS-NEG %s
// CHECK-SYCL-FP-ATOMICS: #define SYCL_USE_NATIVE_FP_ATOMICS
// CHECK-SYCL-FP-ATOMICS-NEG-NOT: #define SYCL_USE_NATIVE_FP_ATOMICS

// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64_fpga-unknown-unknown -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-USM-ADDR-SPACE %s
// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64-unknown-unknown -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-USM-ADDR-SPACE-NEG %s
// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64_gen-unknown-unknown -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-USM-ADDR-SPACE-NEG %s
// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64_x86_64-unknown-unknown -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-USM-ADDR-SPACE-NEG %s
// RUN: %clang_cc1 %s -fsycl-is-device -triple nvptx64-nvidia-nvcl -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-USM-ADDR-SPACE-NEG %s
// CHECK-USM-ADDR-SPACE: #define __ENABLE_USM_ADDR_SPACE__
// CHECK-USM-ADDR-SPACE-NEG-NOT: #define __ENABLE_USM_ADDR_SPACE__

// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64_fpga-unknown-unknown -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-DISABLE-FALLBACK-ASSERT %s
// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64-unknown-unknown -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-DISABLE-FALLBACK-ASSERT-NEG %s
// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64_gen-unknown-unknown -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-DISABLE-FALLBACK-ASSERT-NEG %s
// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64_x86_64-unknown-unknown -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-DISABLE-FALLBACK-ASSERT-NEG %s
// RUN: %clang_cc1 %s -fsycl-is-device -triple nvptx64-nvidia-nvcl -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-DISABLE-FALLBACK-ASSERT-NEG %s
// CHECK-DISABLE-FALLBACK-ASSERT: #define SYCL_DISABLE_FALLBACK_ASSERT
// CHECK-DISABLE-FALLBACK-ASSERT-NEG-NOT: #define SYCL_DISABLE_FALLBACK_ASSERT
