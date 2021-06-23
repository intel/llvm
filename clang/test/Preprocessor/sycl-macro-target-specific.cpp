// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-SYCL-FP-ATOMICS %s
// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64_gen-unknown-unknown-sycldevice -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-SYCL-FP-ATOMICS %s
// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64_x86_64-unknown-unknown-sycldevice -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-SYCL-FP-ATOMICS %s
// RUN: %clang_cc1 %s -fsycl-is-device -triple spir64_fpga-unknown-unknown-sycldevice -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-SYCL-FP-ATOMICS-NEG %s
// RUN: %clang_cc1 %s -fsycl-is-device -triple nvptx64-nvidia-nvcl-sycldevice -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-SYCL-FP-ATOMICS-NEG %s

// CHECK-SYCL-FP-ATOMICS: #define SYCL_USE_NATIVE_FP_ATOMICS
// CHECK-SYCL-FP-ATOMICS-NEG-NOT: #define SYCL_USE_NATIVE_FP_ATOMICS
