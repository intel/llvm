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
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libc --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libc --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libm-fp32 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libc,libm-fp32 --sysroot=%S/Inputs/SYCL -### 2>&1 \
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
/// test behavior of device library link with libm-fp64
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libm-fp64 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_WITH_FP64
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libc,libm-fp64 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_WITH_FP64
// RUN: %clangxx -fsycl %s -fsycl-device-lib=all --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_WITH_FP64
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libc,libm-fp32,libm-fp64 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_WITH_FP64
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libc,all --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_WITH_FP64
// SYCL_DEVICE_LIB_LINK_WITH_FP64: llvm-link{{.*}} "{{.*}}libsycl-crt.bc"
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: "{{.*}}libsycl-complex.bc"
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: "{{.*}}libsycl-complex-fp64.bc"
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: "{{.*}}libsycl-cmath.bc"
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: "{{.*}}libsycl-cmath-fp64.bc"
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: "{{.*}}libsycl-imf.bc"
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: "{{.*}}libsycl-imf-fp64.bc"
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: "{{.*}}libsycl-imf-bf16.bc"
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: "{{.*}}libsycl-fallback-cassert.bc"
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: "{{.*}}libsycl-fallback-cstring.bc"
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: "{{.*}}libsycl-fallback-complex.bc"
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: "{{.*}}libsycl-fallback-complex-fp64.bc"
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: "{{.*}}libsycl-fallback-cmath.bc"
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: "{{.*}}libsycl-fallback-cmath-fp64.bc"
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: "{{.*}}libsycl-fallback-imf.bc"
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: "{{.*}}libsycl-fallback-imf-fp64.bc"
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: "{{.*}}libsycl-fallback-imf-bf16.bc"

/// ###########################################################################
/// test behavior of -fno-sycl-device-lib=libc
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libc --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_LIBC
// SYCL_DEVICE_LIB_LINK_NO_LIBC: llvm-link{{.*}}
// SYCL_DEVICE_LIB_LINK_NO_LIBC-NOT: "{{.*}}libsycl-crt.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBC: "{{.*}}libsycl-complex.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: "{{.*}}libsycl-complex-fp64.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: "{{.*}}libsycl-cmath.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: "{{.*}}libsycl-cmath-fp64.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: "{{.*}}libsycl-imf.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: "{{.*}}libsycl-imf-fp64.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: "{{.*}}libsycl-imf-bf16.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBC-NOT: "{{.*}}libsycl-fallback-cassert.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBC-NOT: "{{.*}}libsycl-fallback-cstring.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: "{{.*}}libsycl-fallback-complex.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: "{{.*}}libsycl-fallback-complex-fp64.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: "{{.*}}libsycl-fallback-cmath.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: "{{.*}}libsycl-fallback-cmath-fp64.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: "{{.*}}libsycl-fallback-imf.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: "{{.*}}libsycl-fallback-imf-fp64.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: "{{.*}}libsycl-fallback-imf-bf16.bc"

/// ###########################################################################
/// test behavior of -fno-sycl-device-lib=libm-fp32,libm-fp64
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libm-fp32,libm-fp64 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_LIBM
// SYCL_DEVICE_LIB_LINK_NO_LIBM: llvm-link{{.*}} "{{.*}}libsycl-crt.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBM-NOT: "{{.*}}libsycl-complex.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBM-NOT: "{{.*}}libsycl-complex-fp64.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBM-NOT: "{{.*}}libsycl-cmath.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBM-NOT: "{{.*}}libsycl-cmath-fp64.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBM-SAME: "{{.*}}libsycl-imf.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBM-SAME: "{{.*}}libsycl-imf-fp64.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBM-SAME: "{{.*}}libsycl-imf-bf16.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBM-SAME: "{{.*}}libsycl-fallback-cassert.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBM-SAME: "{{.*}}libsycl-fallback-cstring.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBM-NOT: "{{.*}}libsycl-fallback-complex.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBM-NOT: "{{.*}}libsycl-fallback-complex-fp64.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBM-NOT: "{{.*}}libsycl-fallback-cmath.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBM-NOT: "{{.*}}libsycl-fallback-cmath-fp64.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBM-SAME: "{{.*}}libsycl-fallback-imf.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBM-SAME: "{{.*}}libsycl-fallback-imf-fp64.bc"
// SYCL_DEVICE_LIB_LINK_NO_LIBM-SAME: "{{.*}}libsycl-fallback-imf-bf16.bc"

/// ###########################################################################
/// test behavior of disabling all device libraries
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libc,libm-fp32 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=all --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libc,all --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libm-fp32,all --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libm-fp64,all --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libc,all,libm-fp64,libm-fp32 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB
// SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB: {{.*}}clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown"
// SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB-NOT: libsycl-cmath.bc

/// ###########################################################################
/// test invalid value for -f[no-]sycl-device-lib
// RUN: not %clangxx -fsycl %s -fsycl-device-lib=libc,dummy --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_INVALID_VALUE -DVal=dummy
// RUN: not %clangxx -fsycl %s -fno-sycl-device-lib=dummy,libm-fp32 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_NO_DEVICE_LIB_INVALID_VALUE -DVal=dummy
// Do separate checks for the compiler-reserved "internal" value
// RUN: not %clangxx -fsycl %s -fsycl-device-lib=internal --sysroot=%S/Inputs/SYCL -### 2>&1		\
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_INVALID_VALUE -DVal=internal
// RUN: not %clangxx -fsycl %s -fno-sycl-device-lib=internal --sysroot=%S/Inputs/SYCL -### 2>&1		\
// RUN:   | FileCheck %s -check-prefix=SYCL_NO_DEVICE_LIB_INVALID_VALUE -DVal=internal
// SYCL_DEVICE_LIB_INVALID_VALUE: error: unsupported argument '[[Val]]' to option '-fsycl-device-lib='
// SYCL_NO_DEVICE_LIB_INVALID_VALUE: error: unsupported argument '[[Val]]' to option '-fno-sycl-device-lib='

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
/// test llvm-link behavior for fno-sycl-device-lib
// RUN: %clangxx -fsycl -fno-sycl-dead-args-optimization -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_LLVM_LINK_NO_DEVICE_LIB
// SYCL_LLVM_LINK_NO_DEVICE_LIB: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-device"
// SYCL_LLVM_LINK_NO_DEVICE_LIB-NOT: llvm-link{{.*}}  "-only-needed"
// SYCL_LLVM_LINK_NO_DEVICE_LIB: sycl-post-link{{.*}} "-spec-const=native" {{.*}} "-o" "{{.*}}.table" "{{.*}}.bc"

/// ###########################################################################
/// test clang-cl behavior for linking sycl-devicelib-host.lib by default
// RUN: %clang_cl -fsycl %s /winsysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_HOST_LIB
// SYCL_DEVICE_HOST_LIB: {{.*}} "--dependent-lib=sycl-devicelib-host" {{.*}}
