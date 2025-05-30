///
/// Perform several driver tests for SYCL device libraries on Windows
///

// REQUIRES: system-windows

/// ###########################################################################

/// test behavior of device library default link
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// SYCL_DEVICE_LIB_LINK_DEFAULT: llvm-link{{.*}} "{{.*}}libsycl-crt.bc"
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: "{{.*}}libsycl-complex.bc"
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: "{{.*}}libsycl-complex-fp64.bc"
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: "{{.*}}libsycl-cmath.bc"
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: "{{.*}}libsycl-cmath-fp64.bc"
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: "{{.*}}libsycl-imf.bc"
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: "{{.*}}libsycl-imf-fp64.bc"
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: "{{.*}}libsycl-imf-bf16.bc"
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: "{{.*}}libsycl-fallback-cassert.bc"
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: "{{.*}}libsycl-fallback-cstring.bc"
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: "{{.*}}libsycl-fallback-complex.bc"
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: "{{.*}}libsycl-fallback-complex-fp64.bc"
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: "{{.*}}libsycl-fallback-cmath.bc"
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: "{{.*}}libsycl-fallback-cmath-fp64.bc"
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: "{{.*}}libsycl-fallback-imf.bc"
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: "{{.*}}libsycl-fallback-imf-fp64.bc"
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: "{{.*}}libsycl-fallback-imf-bf16.bc"

/// ###########################################################################
/// test llvm-link behavior for linking device libraries
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_LLVM_LINK_DEVICE_LIB
// RUN: %clangxx -fsycl -save-temps %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_LLVM_LINK_DEVICE_LIB
// SYCL_LLVM_LINK_DEVICE_LIB: llvm-link{{.*}}  "{{.*}}.bc" "-o" "{{.*}}.bc" "--suppress-warnings"
// SYCL_LLVM_LINK_DEVICE_LIB: llvm-link{{.*}} "-only-needed"
// SYCL_LLVM_LINK_DEVICE_LIB-SAME: "{{.*}}libsycl-crt.bc"
// SYCL_LLVM_LINK_DEVICE_LIB-SAME: "{{.*}}libsycl-complex.bc"
// SYCL_LLVM_LINK_DEVICE_LIB-SAME: "{{.*}}libsycl-complex-fp64.bc"
// SYCL_LLVM_LINK_DEVICE_LIB-SAME: "{{.*}}libsycl-cmath.bc"
// SYCL_LLVM_LINK_DEVICE_LIB-SAME: "{{.*}}libsycl-cmath-fp64.bc"
// SYCL_LLVM_LINK_DEVICE_LIB-SAME: "{{.*}}libsycl-imf.bc"
// SYCL_LLVM_LINK_DEVICE_LIB-SAME: "{{.*}}libsycl-imf-fp64.bc"
// SYCL_LLVM_LINK_DEVICE_LIB-SAME: "{{.*}}libsycl-imf-bf16.bc"
// SYCL_LLVM_LINK_DEVICE_LIB-SAME: "{{.*}}libsycl-fallback-cassert.bc"
// SYCL_LLVM_LINK_DEVICE_LIB-SAME: "{{.*}}libsycl-fallback-cstring.bc"
// SYCL_LLVM_LINK_DEVICE_LIB-SAME: "{{.*}}libsycl-fallback-complex.bc"
// SYCL_LLVM_LINK_DEVICE_LIB-SAME: "{{.*}}libsycl-fallback-complex-fp64.bc"
// SYCL_LLVM_LINK_DEVICE_LIB-SAME: "{{.*}}libsycl-fallback-cmath.bc"
// SYCL_LLVM_LINK_DEVICE_LIB-SAME: "{{.*}}libsycl-fallback-cmath-fp64.bc"
// SYCL_LLVM_LINK_DEVICE_LIB-SAME: "{{.*}}libsycl-fallback-imf.bc"
// SYCL_LLVM_LINK_DEVICE_LIB-SAME: "{{.*}}libsycl-fallback-imf-fp64.bc"
// SYCL_LLVM_LINK_DEVICE_LIB-SAME: "{{.*}}libsycl-fallback-imf-bf16.bc"

/// ###########################################################################
/// test clang-cl behavior for linking sycl-devicelib-host.lib by default
// RUN: %clang_cl -fsycl %s /winsysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_HOST_LIB
// SYCL_DEVICE_HOST_LIB: {{.*}} "--dependent-lib=sycl-devicelib-host" {{.*}}
