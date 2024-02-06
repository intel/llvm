///
/// Perform several driver tests for SYCL device libraries on Linux
///

// UNSUPPORTED: system-windows

/// ###########################################################################

/// test behavior of device library default link and fno-sycl-device-lib-jit-link
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT
// RUN: %clangxx -fsycl -fno-sycl-device-lib-jit-link %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libc --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT
// RUN: %clangxx -fsycl -fno-sycl-device-lib-jit-link %s -fsycl-device-lib=libc --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libm-fp32 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libc,libm-fp32 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
/// ###########################################################################
/// test sycl fallback device libraries are not linked by default
// RUN: %clangxx -fsycl -fsycl-device-lib-jit-link %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_NO_FALLBACK
// SYCL_DEVICE_LIB_NO_FALLBACK: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_NO_FALLBACK-NOT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-{{.*}}.o"
// SYCL_DEVICE_LIB_NO_FALLBACK: llvm-link{{.*}}  "-only-needed"

/// ###########################################################################
/// test behavior of device library link with libm-fp64
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libm-fp64 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libc,libm-fp64 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64
// RUN: %clangxx -fsycl %s -fsycl-device-lib=all --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libc,libm-fp32,libm-fp64 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libc,all --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
/// ###########################################################################

/// test behavior of -fno-sycl-device-lib=libc
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libc --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
/// ###########################################################################

/// test behavior of -fno-sycl-device-lib=libm-fp32,libm-fp64
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libm-fp32,libm-fp64 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBM
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBM: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBM: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBM: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBM: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBM-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBM-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBM: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBM: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBM: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
/// ###########################################################################

/// test behavior of disabling all device libraries
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libc,libm-fp32 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=all --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libc,all --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libm-fp32,all --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libm-fp64,all --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libc,all,libm-fp64,libm-fp32 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB
// SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB: {{.*}}clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB-NEXT: {{.*}}llvm-link{{.*}} {{.*}} "--suppress-warnings"

/// ###########################################################################

/// test invalid value for -f[no-]sycl-device-lib
// RUN: not %clangxx -fsycl %s -fsycl-device-lib=libc,dummy -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_INVALID_VALUE -DVal=dummy
// RUN: not %clangxx -fsycl %s -fno-sycl-device-lib=dummy,libm-fp32 -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_NO_DEVICE_LIB_INVALID_VALUE -DVal=dummy
// Do separate checks for the compiler-reserved "internal" value
// RUN: not %clangxx -fsycl %s -fsycl-device-lib=internal -### 2>&1		\
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_INVALID_VALUE -DVal=internal
// RUN: not %clangxx -fsycl %s -fno-sycl-device-lib=internal -### 2>&1		\
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
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-user-wrappers.o" "-output={{.*}}libsycl-itt-user-wrappers-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-compiler-wrappers.o" "-output={{.*}}libsycl-itt-compiler-wrappers-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-stubs.o" "-output={{.*}}libsycl-itt-stubs-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: llvm-link{{.*}} "-only-needed" "{{.*}}" "-o" "{{.*}}.bc" "--suppress-warnings"

/// ###########################################################################
/// test llvm-link behavior for fno-sycl-device-lib
// RUN: %clangxx -fsycl -fno-sycl-dead-args-optimization -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_LLVM_LINK_NO_DEVICE_LIB
// SYCL_LLVM_LINK_NO_DEVICE_LIB: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-device"
// SYCL_LLVM_LINK_NO_DEVICE_LIB-NOT: llvm-link{{.*}}  "-only-needed"
// SYCL_LLVM_LINK_NO_DEVICE_LIB: sycl-post-link{{.*}}  "-symbols" "-emit-exported-symbols" "-split-esimd" "-lower-esimd" "-O2" "-spec-const=native" "-device-globals" "-o" "{{.*}}.table" "{{.*}}.bc"

/// ###########################################################################
/// test llvm-link behavior for special user input whose filename resembles SYCL device library
// RUN: touch libsycl-crt.o
// RUN: %clangxx -fsycl libsycl-crt.o --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_LLVM_LINK_USER_ONLY_NEEDED
// SYCL_LLVM_LINK_USER_ONLY_NEEDED: llvm-link{{.*}}  "{{.*}}.bc" "-o" "{{.*}}.bc" "--suppress-warnings"
// SYCL_LLVM_LINK_USER_ONLY_NEEDED: llvm-link{{.*}}  "-only-needed" "{{.*}}" "-o" "{{.*}}.bc" "--suppress-warnings"

/// ###########################################################################
/// test llvm-link behavior for linking device libraries
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT
// SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT: llvm-link{{.*}}  "{{.*}}.bc" "-o" "{{.*}}.bc" "--suppress-warnings"
// SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-bfloat16.o" "-output={{.*}}libsycl-fallback-bfloat16-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-itt-user-wrappers.o" "-output={{.*}}libsycl-itt-user-wrappers-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-itt-compiler-wrappers.o" "-output={{.*}}libsycl-itt-compiler-wrappers-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-itt-stubs.o" "-output={{.*}}libsycl-itt-stubs-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB_SPIRV_CPU_AOT-NEXT: llvm-link{{.*}} "-only-needed" "{{.*}}" "-o" "{{.*}}.bc" "--suppress-warnings"

/// ###########################################################################
/// test behavior of libsycl-sanitizer.o linking when -fsanitize=address is available
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL -fsanitize=address -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_SANITIZER
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL -Xsycl-target-frontend -fsanitize=address -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_SANITIZER
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL -Xsycl-target-frontend=spir64 -fsanitize=address -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_SANITIZER
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL -Xarch_device -fsanitize=address -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_SANITIZER
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL -Xarch_device "-fsanitize=address -DUSE_SYCL_DEVICE_ASAN" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_SANITIZER
// SYCL_DEVICE_LIB_SANITIZER: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_SANITIZER-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_SANITIZER-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_SANITIZER-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_SANITIZER-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_SANITIZER-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_SANITIZER-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_SANITIZER-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_SANITIZER-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_SANITIZER-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_SANITIZER-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_SANITIZER-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_SANITIZER-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_SANITIZER-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_SANITIZER-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_SANITIZER-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_SANITIZER-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_SANITIZER-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-user-wrappers.o" "-output={{.*}}libsycl-itt-user-wrappers-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_SANITIZER-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-compiler-wrappers.o" "-output={{.*}}libsycl-itt-compiler-wrappers-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_SANITIZER-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-stubs.o" "-output={{.*}}libsycl-itt-stubs-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_SANITIZER-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-sanitizer.o" "-output={{.*}}libsycl-sanitizer-{{.*}}.o" "-unbundle"
