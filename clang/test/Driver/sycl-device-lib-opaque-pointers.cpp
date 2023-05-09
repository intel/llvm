///
/// Checks that opaque-pointers are being used to llvm-link non SPIR-V devicelib
/// bundles
///
/// FIXME remove once opaque pointers are supported for SPIR-V targets
///

// UNSUPPORTED: system-windows

/// ###########################################################################

// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_OPAQUE_NON_SPIRV
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_OPAQUE_SPIRV
// SYCL_DEVICE_LIB_OPAQUE_NON_SPIRV: llvm-link
// SYCL_DEVICE_LIB_OPAQUE_NON_SPIRV-NOT: "-opaque-pointers"
// SYCL_DEVICE_LIB_OPAQUE_NON_SPIRV: llvm-link{{.*}}"-opaque-pointers"{{.*}}libsycl-crt{{.*}}libsycl-complex{{.*}}libsycl-complex-fp64{{.*}}libsycl-cmath{{.*}}libsycl-cmath-fp64{{.*}}libsycl-imf{{.*}}libsycl-imf-fp64{{.*}}libsycl-imf-bf16{{.*}}libsycl-fallback-cassert{{.*}}libsycl-fallback-cstring{{.*}}libsycl-fallback-complex{{.*}}libsycl-fallback-complex-fp64{{.*}}libsycl-fallback-cmath
// SYCL_DEVICE_LIB_OPAQUE_SPIRV: llvm-link
// SYCL_DEVICE_LIB_OPAQUE_SPIRV-NOT: "-opaque-pointers"
// SYCL_DEVICE_LIB_OPAQUE_SPIRV-NOT: llvm-link{{.*}}"-opaque-pointers"{{.*}}libsycl-crt{{.*}}libsycl-complex{{.*}}libsycl-complex-fp64{{.*}}libsycl-cmath{{.*}}libsycl-cmath-fp64{{.*}}libsycl-imf{{.*}}libsycl-imf-fp64{{.*}}libsycl-imf-bf16{{.*}}libsycl-fallback-cassert{{.*}}libsycl-fallback-cstring{{.*}}libsycl-fallback-complex{{.*}}libsycl-fallback-complex-fp64{{.*}}libsycl-fallback-cmath
// SYCL_DEVICE_LIB_OPAQUE_SPIRV: llvm-link{{.*}}libsycl-crt{{.*}}libsycl-complex{{.*}}libsycl-complex-fp64{{.*}}libsycl-cmath{{.*}}libsycl-cmath-fp64{{.*}}libsycl-imf{{.*}}libsycl-imf-fp64{{.*}}libsycl-imf-bf16{{.*}}libsycl-fallback-cassert{{.*}}libsycl-fallback-cstring{{.*}}libsycl-fallback-complex{{.*}}libsycl-fallback-complex-fp64{{.*}}libsycl-fallback-cmath
/// ###########################################################################
