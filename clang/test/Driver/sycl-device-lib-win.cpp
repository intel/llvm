///
/// Perform several driver tests for SYCL device libraries on Windows
///

// REQUIRES: clang-driver, windows

/// ###########################################################################

/// test behavior of device library default link
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL-windows -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libc --sysroot=%S/Inputs/SYCL-windows -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libm-fp32 --sysroot=%S/Inputs/SYCL-windows -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libc,libm-fp32 --sysroot=%S/Inputs/SYCL-windows -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-crt.obj" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex.obj" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.obj" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath.obj" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.obj" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"

/// ###########################################################################
/// test sycl fallback device libraries are not linked by default
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL-windows -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_NO_FALLBACK
// SYCL_DEVICE_LIB_NO_FALLBACK: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-crt.obj" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_NO_FALLBACK-NOT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-{{.*}}.obj"
// SYCL_DEVICE_LIB_NO_FALLBACK: llvm-link{{.*}}  "-only-needed"

/// ###########################################################################
/// test behavior of device library link with libm-fp64
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libm-fp64 --sysroot=%S/Inputs/SYCL-windows -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libc,libm-fp64 --sysroot=%S/Inputs/SYCL-windows -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64
// RUN: %clangxx -fsycl %s -fsycl-device-lib=all --sysroot=%S/Inputs/SYCL-windows -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libc,libm-fp32,libm-fp64 --sysroot=%S/Inputs/SYCL-windows -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libc,all --sysroot=%S/Inputs/SYCL-windows -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-crt.obj" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex.obj" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.obj" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath.obj" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.obj" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"

/// ###########################################################################

/// test behavior of -fno-sycl-device-lib=libc
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libc --sysroot=%S/Inputs/SYCL-windows -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex.obj" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.obj" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath.obj" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.obj" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
/// ###########################################################################

/// test behavior of -fno-sycl-device-lib=libm-fp32,libm-fp64
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libm-fp32,libm-fp64 --sysroot=%S/Inputs/SYCL-windows -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBM
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBM: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-crt.obj" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
/// ###########################################################################

/// test behavior of disabling all device libraries
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libc,libm-fp32 --sysroot=%S/Inputs/SYCL-windows -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=all --sysroot=%S/Inputs/SYCL-windows -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libc,all --sysroot=%S/Inputs/SYCL-windows -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libm-fp32,all --sysroot=%S/Inputs/SYCL-windows -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libm-fp64,all --sysroot=%S/Inputs/SYCL-windows -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libc,all,libm-fp64,libm-fp32 --sysroot=%S/Inputs/SYCL-windows -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB
// SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB: {{.*}}clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB-NEXT: {{.*}}llvm-link{{.*}} {{.*}} "--suppress-warnings"

/// ###########################################################################

/// test invalid value for -f[no-]sycl-device-lib
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libc,dummy -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_INVALID_VALUE -DVal=dummy
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=dummy,libm-fp32 -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_NO_DEVICE_LIB_INVALID_VALUE -DVal=dummy
// Do separate checks for the compiler-reserved "internal" value
// RUN: %clangxx -fsycl %s -fsycl-device-lib=internal -### 2>&1		\
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_INVALID_VALUE -DVal=internal
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=internal -### 2>&1		\
// RUN:   | FileCheck %s -check-prefix=SYCL_NO_DEVICE_LIB_INVALID_VALUE -DVal=internal
// SYCL_DEVICE_LIB_INVALID_VALUE: error: unsupported argument '[[Val]]' to option 'fsycl-device-lib='
// SYCL_NO_DEVICE_LIB_INVALID_VALUE: error: unsupported argument '[[Val]]' to option 'fno-sycl-device-lib='

/// ###########################################################################
/// test llvm-link behavior for linking device libraries
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL-windows -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_LLVM_LINK_DEVICE_LIB
// RUN: %clangxx -fsycl -save-temps %s --sysroot=%S/Inputs/SYCL-windows -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_LLVM_LINK_DEVICE_LIB
// SYCL_LLVM_LINK_DEVICE_LIB: llvm-link{{.*}}  "{{.*}}.bc" "-o" "{{.*}}.bc" "--suppress-warnings"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-crt.obj" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex.obj" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.obj" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath.obj" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.obj" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: llvm-link{{.*}} "-only-needed" "{{.*}}" "-o" "{{.*}}.bc" "--suppress-warnings"

/// ###########################################################################
/// test llvm-link behavior for fno-sycl-device-lib
// RUN: %clangxx -fsycl -fno-sycl-device-lib=all %s --sysroot=%S/Inputs/SYCL-windows -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_LLVM_LINK_NO_DEVICE_LIB
// SYCL_LLVM_LINK_NO_DEVICE_LIB: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-device"
// SYCL_LLVM_LINK_NO_DEVICE_LIB-NOT: llvm-link{{.*}}  "-only-needed"
// SYCL_LLVM_LINK_NO_DEVICE_LIB: sycl-post-link{{.*}}  "-symbols" "-spec-const=rt" "-o" "{{.*}}.table" "{{.*}}.bc"
