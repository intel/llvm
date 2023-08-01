///
/// Checks that the SYCL libraries modules are first linked with opaque-pointers, 
/// when no GPU Relocatable Device Code mode is used.
///

// UNSUPPORTED: system-windows

/// FIXME: remove once opaque pointers are supported for all SYCL targets (i.e., SPIR-V)

// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fno-gpu-rdc %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN: | FileCheck %s -check-prefix=CHECK_NORDC_OPAQUE_PTR
// CHECK_NORDC_OPAQUE_PTR: clang{{.*}}
// CHECK_NORDC_OPAQUE_PTR-NOT: llvm-link
// CHECK_NORDC_OPAQUE_PTR: clang-offload-bundler
// CHECK_NORDC_OPAQUE_PTR: llvm-link{{.*}}"-only-needed"{{.*}}"-opaque-pointers"{{.*}}libsycl-crt{{.*}}libsycl-complex{{.*}}libsycl-complex-fp64{{.*}}libsycl-cmath{{.*}}libsycl-cmath-fp64{{.*}}libsycl-imf{{.*}}libsycl-imf-fp64{{.*}}libsycl-imf-bf16{{.*}}libsycl-fallback-cassert{{.*}}libsycl-fallback-cstring{{.*}}libsycl-fallback-complex{{.*}}libsycl-fallback-complex-fp64{{.*}}libsycl-fallback-cmath
// CHECK_NORDC_OPAQUE_PTR-NOT: "opaque-pointers"
// CHECK_NORDC_OPAQUE_PTR-NEXT: llvm-link{{.*}}sycl-no-gpu-rdc-opaque-pointers{{.*}}libsycl-crt
// CHECK_NORDC_OPAQUE_PTR-NEXT: sycl-post-link
