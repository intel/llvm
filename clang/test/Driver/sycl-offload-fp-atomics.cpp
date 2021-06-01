/// Check that the macro for enabling native SYCL FP atomics is only passed for
/// targets that offer support for the corresponding SPIR-V extensions.
// RUN: %clangxx -fsycl %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-POS
// RUN: %clangxx -fsycl %s -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-POS
// RUN: %clangxx -fsycl %s -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-POS
// CHECK-POS: clang{{.*}} "-fsycl-is-device"{{.*}} "-DSYCL_USE_NATIVE_FP_ATOMICS"
//
// RUN: %clangxx -fsycl %s -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-NEG
// RUN: %clangxx -fsycl %s -fsycl-targets=nvptx64-nvidia-nvcl-sycldevice -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-NEG
// CHECK-NEG-NOT: -DSYCL_USE_NATIVE_FP_ATOMICS
