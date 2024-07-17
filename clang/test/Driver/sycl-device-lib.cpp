///
/// Perform several driver tests for SYCL device libraries on Linux
///

// UNSUPPORTED: system-windows

/// ###########################################################################

/// test behavior of device library default link and fno-sycl-device-lib-jit-link
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// RUN: %clangxx -fsycl --offload-new-driver -fno-sycl-device-lib-jit-link %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// RUN: %clangxx -fsycl --offload-new-driver %s -fsycl-device-lib=libc --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// RUN: %clangxx -fsycl --offload-new-driver -fno-sycl-device-lib-jit-link %s -fsycl-device-lib=libc --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// RUN: %clangxx -fsycl --offload-new-driver %s -fsycl-device-lib=libm-fp32 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// RUN: %clangxx -fsycl --offload-new-driver %s -fsycl-device-lib=libc,libm-fp32 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// SYCL_DEVICE_LIB_LINK_DEFAULT: clang-linker-wrapper{{.*}} "-sycl-device-libraries=libsycl-crt.new.o
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-complex.new.o
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-complex-fp64.new.o
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-cmath.new.o
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-cmath-fp64.new.o
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-imf.new.o
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-imf-fp64.new.o
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-imf-bf16.new.o
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-fallback-cassert.new.o
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-fallback-cstring.new.o
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-fallback-complex.new.o
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-fallback-complex-fp64.new.o
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-fallback-cmath.new.o
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-fallback-cmath-fp64.new.o
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-fallback-imf.new.o
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-fallback-imf-fp64.new.o
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-fallback-imf-bf16.new.o

/// ###########################################################################
/// test sycl fallback device libraries are not linked by default
// RUN: %clangxx -fsycl --offload-new-driver -fsycl-device-lib-jit-link %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_NO_FALLBACK
// SYCL_DEVICE_LIB_NO_FALLBACK: clang-linker-wrapper{{.*}} "-sycl-device-libraries=libsycl-crt.new.o
// SYCL_DEVICE_LIB_NO_FALLBACK-NOT: libsycl-fallback

/// ###########################################################################
/// test behavior of device library link with libm-fp64
// RUN: %clangxx -fsycl --offload-new-driver %s -fsycl-device-lib=libm-fp64 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_WITH_FP64
// RUN: %clangxx -fsycl --offload-new-driver %s -fsycl-device-lib=libc,libm-fp64 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_WITH_FP64
// RUN: %clangxx -fsycl --offload-new-driver %s -fsycl-device-lib=all --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_WITH_FP64
// RUN: %clangxx -fsycl --offload-new-driver %s -fsycl-device-lib=libc,libm-fp32,libm-fp64 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_WITH_FP64
// RUN: %clangxx -fsycl --offload-new-driver %s -fsycl-device-lib=libc,all --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_WITH_FP64
// SYCL_DEVICE_LIB_LINK_WITH_FP64: clang-linker-wrapper{{.*}} "-sycl-device-libraries=libsycl-crt.new.o
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: {{.*}}libsycl-complex.new.o
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: {{.*}}libsycl-complex-fp64.new.o
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: {{.*}}libsycl-cmath.new.o
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: {{.*}}libsycl-cmath-fp64.new.o
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: {{.*}}libsycl-imf.new.o
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: {{.*}}libsycl-imf-fp64.new.o
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: {{.*}}libsycl-imf-bf16.new.o
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: {{.*}}libsycl-fallback-cassert.new.o
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: {{.*}}libsycl-fallback-cstring.new.o
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: {{.*}}libsycl-fallback-complex.new.o
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: {{.*}}libsycl-fallback-complex-fp64.new.o
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: {{.*}}libsycl-fallback-cmath.new.o
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: {{.*}}libsycl-fallback-cmath-fp64.new.o
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: {{.*}}libsycl-fallback-imf.new.o
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: {{.*}}libsycl-fallback-imf-fp64.new.o
// SYCL_DEVICE_LIB_LINK_WITH_FP64-SAME: {{.*}}libsycl-fallback-imf-bf16.new.o
/// ###########################################################################

/// test behavior of -fno-sycl-device-lib=libc
// RUN: %clangxx -fsycl --offload-new-driver %s -fno-sycl-device-lib=libc --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_LIBC
// SYCL_DEVICE_LIB_LINK_NO_LIBC: clang-linker-wrapper{{.*}} "-sycl-device-libraries
// SYCL_DEVICE_LIB_LINK_NO_LIBC-NOT: {{.*}}libsycl-crt.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBC: {{.*}}libsycl-complex.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: {{.*}}libsycl-complex-fp64.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: {{.*}}libsycl-cmath.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: {{.*}}libsycl-cmath-fp64.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: {{.*}}libsycl-imf.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: {{.*}}libsycl-imf-fp64.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: {{.*}}libsycl-imf-bf16.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBC-NOT: {{.*}}libsycl-fallback-cassert.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBC-NOT: {{.*}}libsycl-fallback-cstring.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: {{.*}}libsycl-fallback-complex.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: {{.*}}libsycl-fallback-complex-fp64.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: {{.*}}libsycl-fallback-cmath.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: {{.*}}libsycl-fallback-cmath-fp64.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: {{.*}}libsycl-fallback-imf.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: {{.*}}libsycl-fallback-imf-fp64.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBC-SAME: {{.*}}libsycl-fallback-imf-bf16.new.o
/// ###########################################################################

/// test behavior of -fno-sycl-device-lib=libm-fp32,libm-fp64
// RUN: %clangxx -fsycl --offload-new-driver %s -fno-sycl-device-lib=libm-fp32,libm-fp64 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_LIBM
// SYCL_DEVICE_LIB_LINK_NO_LIBM: clang-linker-wrapper{{.*}} "-sycl-device-libraries
// SYCL_DEVICE_LIB_LINK_NO_LIBM: {{.*}}libsycl-crt.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBM-NOT: {{.*}}libsycl-complex.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBM-NOT: {{.*}}libsycl-complex-fp64.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBM-NOT: {{.*}}libsycl-cmath.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBM-NOT: {{.*}}libsycl-cmath-fp64.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBM-SAME: {{.*}}libsycl-imf.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBM-SAME: {{.*}}libsycl-imf-fp64.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBM-SAME: {{.*}}libsycl-imf-bf16.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBM-SAME: {{.*}}libsycl-fallback-cassert.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBM-SAME: {{.*}}libsycl-fallback-cstring.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBM-NOT: {{.*}}libsycl-fallback-complex.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBM-NOT: {{.*}}libsycl-fallback-complex-fp64.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBM-NOT: {{.*}}libsycl-fallback-cmath.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBM-NOT: {{.*}}libsycl-fallback-cmath-fp64.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBM-SAME: {{.*}}libsycl-fallback-imf.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBM-SAME: {{.*}}libsycl-fallback-imf-fp64.new.o
// SYCL_DEVICE_LIB_LINK_NO_LIBM-SAME: {{.*}}libsycl-fallback-imf-bf16.new.o
/// ###########################################################################

/// test behavior of disabling all device libraries
// RUN: %clangxx -fsycl --offload-new-driver %s -fno-sycl-device-lib=libc,libm-fp32 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB
// RUN: %clangxx -fsycl --offload-new-driver %s -fno-sycl-device-lib=all --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB
// RUN: %clangxx -fsycl --offload-new-driver %s -fno-sycl-device-lib=libc,all --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB
// RUN: %clangxx -fsycl --offload-new-driver %s -fno-sycl-device-lib=libm-fp32,all --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB
// RUN: %clangxx -fsycl --offload-new-driver %s -fno-sycl-device-lib=libm-fp64,all --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB
// RUN: %clangxx -fsycl --offload-new-driver %s -fno-sycl-device-lib=libc,all,libm-fp64,libm-fp32 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB
// SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB: {{.*}}clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown"
// SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB-NOT: libsycl-cmath.new.o

/// ###########################################################################

/// test invalid value for -f[no-]sycl-device-lib
// RUN: not %clangxx -fsycl --offload-new-driver %s -fsycl-device-lib=libc,dummy -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_INVALID_VALUE -DVal=dummy
// RUN: not %clangxx -fsycl --offload-new-driver %s -fno-sycl-device-lib=dummy,libm-fp32 -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_NO_DEVICE_LIB_INVALID_VALUE -DVal=dummy
// Do separate checks for the compiler-reserved "internal" value
// RUN: not %clangxx -fsycl --offload-new-driver %s -fsycl-device-lib=internal -### 2>&1		\
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_INVALID_VALUE -DVal=internal
// RUN: not %clangxx -fsycl --offload-new-driver %s -fno-sycl-device-lib=internal -### 2>&1		\
// RUN:   | FileCheck %s -check-prefix=SYCL_NO_DEVICE_LIB_INVALID_VALUE -DVal=internal
// SYCL_DEVICE_LIB_INVALID_VALUE: error: unsupported argument '[[Val]]' to option '-fsycl-device-lib='
// SYCL_NO_DEVICE_LIB_INVALID_VALUE: error: unsupported argument '[[Val]]' to option '-fno-sycl-device-lib='

/// ###########################################################################
/// test behavior of libsycl-sanitizer.o linking when -fsanitize=address is available
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -fsanitize=address -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_SANITIZER
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xsycl-target-frontend -fsanitize=address -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_SANITIZER
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xsycl-target-frontend=spir64 -fsanitize=address -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_SANITIZER
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device -fsanitize=address -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_SANITIZER
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device "-fsanitize=address -DUSE_SYCL_DEVICE_ASAN" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_SANITIZER
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device "-fsanitize=address -DUSE_SYCL_DEVICE_ASAN" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_ASAN_MACRO
// SYCL_DEVICE_LIB_SANITIZER: clang-linker-wrapper{{.*}} "-sycl-device-libraries
// SYCL_DEVICE_LIB_SANITIZER: {{.*}}libsycl-crt.new.o
// SYCL_DEVICE_LIB_SANITIZER-SAME: {{.*}}libsycl-complex.
// SYCL_DEVICE_LIB_SANITIZER-SAME: {{.*}}libsycl-complex-fp64.
// SYCL_DEVICE_LIB_SANITIZER-SAME: {{.*}}libsycl-cmath.new.o
// SYCL_DEVICE_LIB_SANITIZER-SAME: {{.*}}libsycl-cmath-fp64.new.o
// SYCL_DEVICE_LIB_SANITIZER-SAME: {{.*}}libsycl-imf.new.o
// SYCL_DEVICE_LIB_SANITIZER-SAME: {{.*}}libsycl-imf-fp64.new.o
// SYCL_DEVICE_LIB_SANITIZER-SAME: {{.*}}libsycl-imf-bf16.new.o
// SYCL_DEVICE_LIB_SANITIZER-SAME: {{.*}}libsycl-fallback-cassert.new.o
// SYCL_DEVICE_LIB_SANITIZER-SAME: {{.*}}libsycl-fallback-cstring.new.o
// SYCL_DEVICE_LIB_SANITIZER-SAME: {{.*}}libsycl-fallback-complex.new.o
// SYCL_DEVICE_LIB_SANITIZER-SAME: {{.*}}libsycl-fallback-complex-fp64.new.o
// SYCL_DEVICE_LIB_SANITIZER-SAME: {{.*}}libsycl-fallback-cmath.new.o
// SYCL_DEVICE_LIB_SANITIZER-SAME: {{.*}}libsycl-fallback-cmath-fp64.new.o
// SYCL_DEVICE_LIB_SANITIZER-SAME: {{.*}}libsycl-fallback-imf.new.o
// SYCL_DEVICE_LIB_SANITIZER-SAME: {{.*}}libsycl-fallback-imf-fp64.new.o
// SYCL_DEVICE_LIB_SANITIZER-SAME: {{.*}}libsycl-fallback-imf-bf16.new.o
// SYCL_DEVICE_LIB_SANITIZER-SAME: {{.*}}libsycl-sanitizer.new.o
// SYCL_DEVICE_ASAN_MACRO: "-cc1"
// SYCL_DEVICE_ASAN_MACRO-SAME: "USE_SYCL_DEVICE_ASAN"
// SYCL_DEVICE_ASAN_MACRO: libsycl-sanitizer.new.o
