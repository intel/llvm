///
/// Perform several driver tests for SYCL device libraries on Windows
///
// REQUIRES: clang-driver, windows

/// ###########################################################################

/// test behavior of device library default link
// RUN: %clangxx -fsycl %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libc -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libm-fp32 -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libc,libm-fp32 -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libm-fp64 -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-msvc.o" "-outputs={{.*}}libsycl-msvc-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-complex.o" "-outputs={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-cmath.o" "-outputs={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-fallback-cassert.o" "-outputs={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-fallback-complex.o" "-outputs={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-fallback-cmath.o" "-outputs={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"

/// ###########################################################################
/// test behavior of device library link with libm-fp64
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libm-fp64 -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libc,libm-fp64 -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64
// RUN: %clangxx -fsycl %s -fsycl-device-lib=all -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libc,libm-fp32,libm-fp64 -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libc,all -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-msvc.o" "-outputs={{.*}}libsycl-msvc-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-complex.o" "-outputs={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-complex-fp64.o" "-outputs={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-cmath.o" "-outputs={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-cmath-fp64.o" "-outputs={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-fallback-cassert.o" "-outputs={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-fallback-complex.o" "-outputs={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-fallback-complex-fp64.o" "-outputs={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-fallback-cmath.o" "-outputs={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_WITH_FP64-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-fallback-cmath-fp64.o" "-outputs={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"

/// ###########################################################################

/// test behavior of -fno-sycl-device-lib=libc
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libc -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-complex.o" "-outputs={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-cmath.o" "-outputs={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-fallback-complex.o" "-outputs={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBC-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-fallback-cmath.o" "-outputs={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"

/// ###########################################################################

/// test behavior of -fno-sycl-device-lib=libm-fp32
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libm-fp32 -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBM_FP32
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBM_FP32: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-msvc.o" "-outputs={{.*}}libsycl-msvc-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_LIBM_FP32-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-fallback-cassert.o" "-outputs={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"

/// ###########################################################################

/// test behavior of disabling all device libraries
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libc,libm-fp32 -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=all -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libc,all -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libm-fp32,all -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libm-fp64,all -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=libc,all,libm-fp64,libm-fp32 -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB
// SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB: {{.*}}clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown-sycldevice"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB-NEXT: {{.*}}llvm-link{{.*}} {{.*}} "--suppress-warnings"

/// ###########################################################################

/// test invalid value for -f[no-]sycl-device-lib
// RUN: %clangxx -fsycl %s -fsycl-device-lib=libc,dummy -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_INVALID_VALUE
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib=dummy,libm-fp32 -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_NO_DEVICE_LIB_INVALID_VALUE
// SYCL_DEVICE_LIB_INVALID_VALUE: error: unsupported argument 'dummy' to option 'fsycl-device-lib='
// SYCL_NO_DEVICE_LIB_INVALID_VALUE: error: unsupported argument 'dummy' to option 'fno-sycl-device-lib='
