///
/// Perform several driver tests for SYCL device libraries on Linux
///

// UNSUPPORTED: system-windows

/// ###########################################################################

/// test behavior of device library default link and fno-sycl-device-lib-jit-link
// RUN: %clangxx -fsycl --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// RUN: %clangxx -fsycl --no-offload-new-driver -fno-sycl-device-lib-jit-link %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// RUN: %clangxx -fsycl --no-offload-new-driver %s -fsycl-device-lib=libc --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// RUN: %clangxx -fsycl --no-offload-new-driver -fno-sycl-device-lib-jit-link %s -fsycl-device-lib=libc --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// RUN: %clangxx -fsycl --no-offload-new-driver %s -fsycl-device-lib=libm-fp32 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// RUN: %clangxx -fsycl --no-offload-new-driver %s -fsycl-device-lib=libc,libm-fp32 --sysroot=%S/Inputs/SYCL -### 2>&1 \
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
/// test sycl fallback device libraries are not linked by default
// RUN: %clangxx -fsycl --no-offload-new-driver -fsycl-device-lib-jit-link %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_NO_FALLBACK
// SYCL_DEVICE_LIB_NO_FALLBACK: llvm-link{{.*}} "{{.*}}libsycl-crt.bc"
// SYCL_DEVICE_LIB_NO_FALLBACK-NOT: "{{.*}}libsycl-fallback-{{.*}}.bc"

/// ###########################################################################
/// test behavior of device library link with libm-fp64
// RUN: %clangxx -fsycl --no-offload-new-driver %s -fsycl-device-lib=libm-fp64 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_WITH_FP64
// RUN: %clangxx -fsycl --no-offload-new-driver %s -fsycl-device-lib=libc,libm-fp64 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_WITH_FP64
// RUN: %clangxx -fsycl --no-offload-new-driver %s -fsycl-device-lib=all --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_WITH_FP64
// RUN: %clangxx -fsycl --no-offload-new-driver %s -fsycl-device-lib=libc,libm-fp32,libm-fp64 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_WITH_FP64
// RUN: %clangxx -fsycl --no-offload-new-driver %s -fsycl-device-lib=libc,all --sysroot=%S/Inputs/SYCL -### 2>&1 \
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
// RUN: %clangxx -fsycl --no-offload-new-driver %s -fno-sycl-device-lib=libc --sysroot=%S/Inputs/SYCL -### 2>&1 \
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
// RUN: %clangxx -fsycl --no-offload-new-driver %s -fno-sycl-device-lib=libm-fp32,libm-fp64 --sysroot=%S/Inputs/SYCL -### 2>&1 \
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
// RUN: %clangxx -fsycl --no-offload-new-driver %s -fno-sycl-device-lib=libc,libm-fp32 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB
// RUN: %clangxx -fsycl --no-offload-new-driver %s -fno-sycl-device-lib=all --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB
// RUN: %clangxx -fsycl --no-offload-new-driver %s -fno-sycl-device-lib=libc,all --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB
// RUN: %clangxx -fsycl --no-offload-new-driver %s -fno-sycl-device-lib=libm-fp32,all --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB
// RUN: %clangxx -fsycl --no-offload-new-driver %s -fno-sycl-device-lib=libm-fp64,all --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB
// RUN: %clangxx -fsycl --no-offload-new-driver %s -fno-sycl-device-lib=libc,all,libm-fp64,libm-fp32 --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB
// SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB: {{.*}}clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown"
// SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB-NOT: libsycl-cmath.bc

/// ###########################################################################

/// test invalid value for -f[no-]sycl-device-lib
// RUN: not %clangxx -fsycl --no-offload-new-driver %s -fsycl-device-lib=libc,dummy -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_INVALID_VALUE -DVal=dummy
// RUN: not %clangxx -fsycl --no-offload-new-driver %s -fno-sycl-device-lib=dummy,libm-fp32 -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_NO_DEVICE_LIB_INVALID_VALUE -DVal=dummy
// Do separate checks for the compiler-reserved "internal" value
// RUN: not %clangxx -fsycl --no-offload-new-driver %s -fsycl-device-lib=internal -### 2>&1		\
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_INVALID_VALUE -DVal=internal
// RUN: not %clangxx -fsycl --no-offload-new-driver %s -fno-sycl-device-lib=internal -### 2>&1		\
// RUN:   | FileCheck %s -check-prefix=SYCL_NO_DEVICE_LIB_INVALID_VALUE -DVal=internal
// SYCL_DEVICE_LIB_INVALID_VALUE: error: unsupported argument '[[Val]]' to option '-fsycl-device-lib='
// SYCL_NO_DEVICE_LIB_INVALID_VALUE: error: unsupported argument '[[Val]]' to option '-fno-sycl-device-lib='

/// ###########################################################################
/// test llvm-link behavior for linking device libraries
// RUN: %clangxx -fsycl --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_LLVM_LINK_DEVICE_LIB
// RUN: %clangxx -fsycl --no-offload-new-driver -save-temps %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_LLVM_LINK_DEVICE_LIB
// RUN: %clangxx -fsycl --no-offload-new-driver -fsycl-targets=spir64_x86_64 %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
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
// RUN: %clangxx -fsycl --no-offload-new-driver -fno-sycl-dead-args-optimization -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_LLVM_LINK_NO_DEVICE_LIB
// SYCL_LLVM_LINK_NO_DEVICE_LIB: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-device"
// SYCL_LLVM_LINK_NO_DEVICE_LIB-NOT: llvm-link{{.*}}  "-only-needed"
// SYCL_LLVM_LINK_NO_DEVICE_LIB: sycl-post-link{{.*}}  "-symbols" "-emit-exported-symbols" "-emit-imported-symbols"{{.*}} "-o" "{{.*}}.table" "{{.*}}.bc"

/// ###########################################################################
/// test llvm-link behavior for special user input whose filename resembles SYCL device library
// RUN: touch libsycl-crt.o
// RUN: %clangxx -fsycl --no-offload-new-driver libsycl-crt.o --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_LLVM_LINK_USER_ONLY_NEEDED
// SYCL_LLVM_LINK_USER_ONLY_NEEDED: llvm-link{{.*}}  "{{.*}}.bc" "-o" "{{.*}}.bc" "--suppress-warnings"
// SYCL_LLVM_LINK_USER_ONLY_NEEDED: llvm-link{{.*}}  "-only-needed" "{{.*}}" "-o" "{{.*}}.bc" "--suppress-warnings"

/// ###########################################################################
/// test behavior of libsycl-asan.bc linking when -fsanitize=address is available
// RUN: %clangxx -fsycl --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL -fsanitize=address -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN
// RUN: %clangxx -fsycl --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xsycl-target-frontend -fsanitize=address -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN
// RUN: %clangxx -fsycl --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xsycl-target-frontend=spir64 -fsanitize=address -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN
// RUN: %clangxx -fsycl --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device -fsanitize=address -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN
// RUN: %clangxx -fsycl --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device "-fsanitize=address -DUSE_SYCL_DEVICE_ASAN" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN
// RUN: %clangxx -fsycl --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device "-fsanitize=address -DUSE_SYCL_DEVICE_ASAN" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_ASAN_MACRO
// SYCL_DEVICE_LIB_ASAN: llvm-link{{.*}} "{{.*}}libsycl-crt.bc"
// SYCL_DEVICE_LIB_ASAN-SAME: "{{.*}}libsycl-complex.bc"
// SYCL_DEVICE_LIB_ASAN-SAME: "{{.*}}libsycl-complex-fp64.bc"
// SYCL_DEVICE_LIB_ASAN-SAME: "{{.*}}libsycl-cmath.bc"
// SYCL_DEVICE_LIB_ASAN-SAME: "{{.*}}libsycl-cmath-fp64.bc"
// SYCL_DEVICE_LIB_ASAN-SAME: "{{.*}}libsycl-imf.bc"
// SYCL_DEVICE_LIB_ASAN-SAME: "{{.*}}libsycl-imf-fp64.bc"
// SYCL_DEVICE_LIB_ASAN-SAME: "{{.*}}libsycl-imf-bf16.bc"
// SYCL_DEVICE_LIB_ASAN-SAME: "{{.*}}libsycl-fallback-cassert.bc"
// SYCL_DEVICE_LIB_ASAN-SAME: "{{.*}}libsycl-fallback-cstring.bc"
// SYCL_DEVICE_LIB_ASAN-SAME: "{{.*}}libsycl-fallback-complex.bc"
// SYCL_DEVICE_LIB_ASAN-SAME: "{{.*}}libsycl-fallback-complex-fp64.bc"
// SYCL_DEVICE_LIB_ASAN-SAME: "{{.*}}libsycl-fallback-cmath.bc"
// SYCL_DEVICE_LIB_ASAN-SAME: "{{.*}}libsycl-fallback-cmath-fp64.bc"
// SYCL_DEVICE_LIB_ASAN-SAME: "{{.*}}libsycl-fallback-imf.bc"
// SYCL_DEVICE_LIB_ASAN-SAME: "{{.*}}libsycl-fallback-imf-fp64.bc"
// SYCL_DEVICE_LIB_ASAN-SAME: "{{.*}}libsycl-fallback-imf-bf16.bc"
// SYCL_DEVICE_LIB_ASAN-SAME: "{{.*}}libsycl-asan.bc"
// SYCL_DEVICE_ASAN_MACRO: "-cc1"
// SYCL_DEVICE_ASAN_MACRO-SAME: "USE_SYCL_DEVICE_ASAN"
// SYCL_DEVICE_ASAN_MACRO: llvm-link{{.*}} "-only-needed"
// SYCL_DEVICE_ASAN_MACRO-SAME: "{{.*}}libsycl-asan.bc"

/// ###########################################################################
/// test behavior of linking libsycl-asan-pvc for PVC target AOT compilation when asan flag is applied.
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=address -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_PVC
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device pvc" --no-offload-new-driver %s \
// RUN: --sysroot=%S/Inputs/SYCL -Xarch_device -fsanitize=address -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_PVC
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device pvc" --no-offload-new-driver %s \
// RUN: --sysroot=%S/Inputs/SYCL -Xarch_device -fsanitize=address -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_PVC
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device 12.60.7" --no-offload-new-driver %s \
// RUN: --sysroot=%S/Inputs/SYCL -Xarch_device -fsanitize=address -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_PVC
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xs "-device 12.60.7" --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=address -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_PVC
// SYCL_DEVICE_LIB_ASAN_PVC: llvm-link{{.*}} "{{.*}}libsycl-crt.bc"
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: "{{.*}}libsycl-complex.bc"
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: "{{.*}}libsycl-complex-fp64.bc"
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: "{{.*}}libsycl-cmath.bc"
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: "{{.*}}libsycl-cmath-fp64.bc"
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: "{{.*}}libsycl-imf.bc"
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: "{{.*}}libsycl-imf-fp64.bc"
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: "{{.*}}libsycl-imf-bf16.bc"
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: "{{.*}}libsycl-fallback-cassert.bc"
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: "{{.*}}libsycl-fallback-cstring.bc"
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: "{{.*}}libsycl-fallback-complex.bc"
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: "{{.*}}libsycl-fallback-complex-fp64.bc"
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: "{{.*}}libsycl-fallback-cmath.bc"
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: "{{.*}}libsycl-fallback-cmath-fp64.bc"
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: "{{.*}}libsycl-fallback-imf.bc"
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: "{{.*}}libsycl-fallback-imf-fp64.bc"
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: "{{.*}}libsycl-fallback-imf-bf16.bc"
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: "{{.*}}libsycl-asan-pvc.bc"

/// ###########################################################################
/// test behavior of linking libsycl-asan-cpu for CPU target AOT compilation when asan flag is applied.
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=address -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_CPU
// SYCL_DEVICE_LIB_ASAN_CPU: llvm-link{{.*}} "{{.*}}libsycl-crt.bc"
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: "{{.*}}libsycl-complex.bc"
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: "{{.*}}libsycl-complex-fp64.bc"
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: "{{.*}}libsycl-cmath.bc"
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: "{{.*}}libsycl-cmath-fp64.bc"
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: "{{.*}}libsycl-imf.bc"
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: "{{.*}}libsycl-imf-fp64.bc"
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: "{{.*}}libsycl-imf-bf16.bc"
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: "{{.*}}libsycl-fallback-cassert.bc"
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: "{{.*}}libsycl-fallback-cstring.bc"
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: "{{.*}}libsycl-fallback-complex.bc"
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: "{{.*}}libsycl-fallback-complex-fp64.bc"
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: "{{.*}}libsycl-fallback-cmath.bc"
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: "{{.*}}libsycl-fallback-cmath-fp64.bc"
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: "{{.*}}libsycl-fallback-imf.bc"
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: "{{.*}}libsycl-fallback-imf-fp64.bc"
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: "{{.*}}libsycl-fallback-imf-bf16.bc"
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: "{{.*}}libsycl-asan-cpu.bc"

/// ###########################################################################
/// test behavior of linking libsycl-asan-dg2 for DG2 target AOT compilation when asan flag is applied.
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_dg2_g10 --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=address -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_DG2
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device dg2" --no-offload-new-driver %s \
// RUN: --sysroot=%S/Inputs/SYCL -Xarch_device -fsanitize=address -### 2>&1 \
// RUN: | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_DG2
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device dg2" --no-offload-new-driver %s \
// RUN: --sysroot=%S/Inputs/SYCL -Xarch_device -fsanitize=address -### 2>&1 \
// RUN: | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_DG2
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xs "-device dg2" --no-offload-new-driver %s \
// RUN: --sysroot=%S/Inputs/SYCL -Xarch_device -fsanitize=address -### 2>&1 \
// RUN: | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_DG2
// SYCL_DEVICE_LIB_ASAN_DG2: llvm-link{{.*}} "{{.*}}libsycl-crt.bc"
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: "{{.*}}libsycl-complex.bc"
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: "{{.*}}libsycl-complex-fp64.bc"
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: "{{.*}}libsycl-cmath.bc"
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: "{{.*}}libsycl-cmath-fp64.bc"
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: "{{.*}}libsycl-imf.bc"
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: "{{.*}}libsycl-imf-fp64.bc"
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: "{{.*}}libsycl-imf-bf16.bc"
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: "{{.*}}libsycl-fallback-cassert.bc"
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: "{{.*}}libsycl-fallback-cstring.bc"
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: "{{.*}}libsycl-fallback-complex.bc"
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: "{{.*}}libsycl-fallback-complex-fp64.bc"
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: "{{.*}}libsycl-fallback-cmath.bc"
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: "{{.*}}libsycl-fallback-cmath-fp64.bc"
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: "{{.*}}libsycl-fallback-imf.bc"
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: "{{.*}}libsycl-fallback-imf-fp64.bc"
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: "{{.*}}libsycl-fallback-imf-bf16.bc"
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: "{{.*}}libsycl-asan-dg2.bc"


/// ###########################################################################
/// test behavior of libsycl-msan.bc linking when -fsanitize=memory is available
// RUN: %clangxx -fsycl --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL -fsanitize=memory -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_MSAN
// RUN: %clangxx -fsycl --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xsycl-target-frontend -fsanitize=memory -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_MSAN
// RUN: %clangxx -fsycl --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xsycl-target-frontend=spir64 -fsanitize=memory -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_MSAN
// RUN: %clangxx -fsycl --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device -fsanitize=memory -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_MSAN
// RUN: %clangxx -fsycl --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device "-fsanitize=memory -DUSE_SYCL_DEVICE_MSAN" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_MSAN
// RUN: %clangxx -fsycl --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device "-fsanitize=memory -DUSE_SYCL_DEVICE_MSAN" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_MSAN_MACRO
// SYCL_DEVICE_LIB_MSAN: llvm-link{{.*}} "{{.*}}libsycl-crt.bc"
// SYCL_DEVICE_LIB_MSAN-SAME: "{{.*}}libsycl-complex.bc"
// SYCL_DEVICE_LIB_MSAN-SAME: "{{.*}}libsycl-complex-fp64.bc"
// SYCL_DEVICE_LIB_MSAN-SAME: "{{.*}}libsycl-cmath.bc"
// SYCL_DEVICE_LIB_MSAN-SAME: "{{.*}}libsycl-cmath-fp64.bc"
// SYCL_DEVICE_LIB_MSAN-SAME: "{{.*}}libsycl-imf.bc"
// SYCL_DEVICE_LIB_MSAN-SAME: "{{.*}}libsycl-imf-fp64.bc"
// SYCL_DEVICE_LIB_MSAN-SAME: "{{.*}}libsycl-imf-bf16.bc"
// SYCL_DEVICE_LIB_MSAN-SAME: "{{.*}}libsycl-fallback-cassert.bc"
// SYCL_DEVICE_LIB_MSAN-SAME: "{{.*}}libsycl-fallback-cstring.bc"
// SYCL_DEVICE_LIB_MSAN-SAME: "{{.*}}libsycl-fallback-complex.bc"
// SYCL_DEVICE_LIB_MSAN-SAME: "{{.*}}libsycl-fallback-complex-fp64.bc"
// SYCL_DEVICE_LIB_MSAN-SAME: "{{.*}}libsycl-fallback-cmath.bc"
// SYCL_DEVICE_LIB_MSAN-SAME: "{{.*}}libsycl-fallback-cmath-fp64.bc"
// SYCL_DEVICE_LIB_MSAN-SAME: "{{.*}}libsycl-fallback-imf.bc"
// SYCL_DEVICE_LIB_MSAN-SAME: "{{.*}}libsycl-fallback-imf-fp64.bc"
// SYCL_DEVICE_LIB_MSAN-SAME: "{{.*}}libsycl-fallback-imf-bf16.bc"
// SYCL_DEVICE_LIB_MSAN-SAME: "{{.*}}libsycl-msan.bc"
// SYCL_DEVICE_MSAN_MACRO: "-cc1"
// SYCL_DEVICE_MSAN_MACRO-SAME: "USE_SYCL_DEVICE_MSAN"
// SYCL_DEVICE_MSAN_MACRO: llvm-link{{.*}} "-only-needed"
// SYCL_DEVICE_MSAN_MACRO-SAME: "{{.*}}libsycl-msan.bc"

/// ###########################################################################
/// test behavior of linking libsycl-msan-pvc for PVC target AOT compilation when msan flag is applied.
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=memory -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_MSAN_PVC
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device pvc" --no-offload-new-driver %s \
// RUN: --sysroot=%S/Inputs/SYCL -Xarch_device -fsanitize=memory -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_MSAN_PVC
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device pvc" --no-offload-new-driver %s \
// RUN: --sysroot=%S/Inputs/SYCL -Xarch_device -fsanitize=memory -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_MSAN_PVC
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device 12.60.7" --no-offload-new-driver %s \
// RUN: --sysroot=%S/Inputs/SYCL -Xarch_device -fsanitize=memory -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_MSAN_PVC
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xs "-device 12.60.7" --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=memory -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_MSAN_PVC
// SYCL_DEVICE_LIB_MSAN_PVC: llvm-link{{.*}} "-only-needed" "{{.*}}libsycl-crt.bc"
// SYCL_DEVICE_LIB_MSAN_PVC-SAME: "{{.*}}libsycl-msan-pvc.bc"


/// ###########################################################################
/// test behavior of linking libsycl-msan-cpu for CPU target AOT compilation when msan flag is applied.
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=memory -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_MSAN_CPU
// SYCL_DEVICE_LIB_MSAN_CPU: llvm-link{{.*}} "-only-needed" "{{.*}}libsycl-crt.bc"
// SYCL_DEVICE_LIB_MSAN_CPU-SAME: "{{.*}}libsycl-msan-cpu.bc"

/// ###########################################################################
/// test behavior of libsycl-tsan.bc linking when -fsanitize=thread is available
// RUN: %clangxx -fsycl --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL -fsanitize=thread -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_TSAN
// RUN: %clangxx -fsycl --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xsycl-target-frontend -fsanitize=thread -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_TSAN
// RUN: %clangxx -fsycl --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xsycl-target-frontend=spir64 -fsanitize=thread -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_TSAN
// RUN: %clangxx -fsycl --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device -fsanitize=thread -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_TSAN
// RUN: %clangxx -fsycl --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device "-fsanitize=thread -DUSE_SYCL_DEVICE_TSAN" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_TSAN
// RUN: %clangxx -fsycl --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device "-fsanitize=thread -DUSE_SYCL_DEVICE_TSAN" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_TSAN_MACRO
// SYCL_DEVICE_LIB_TSAN: llvm-link{{.*}} "{{.*}}libsycl-crt.bc"
// SYCL_DEVICE_LIB_TSAN-SAME: "{{.*}}libsycl-complex.bc"
// SYCL_DEVICE_LIB_TSAN-SAME: "{{.*}}libsycl-complex-fp64.bc"
// SYCL_DEVICE_LIB_TSAN-SAME: "{{.*}}libsycl-cmath.bc"
// SYCL_DEVICE_LIB_TSAN-SAME: "{{.*}}libsycl-cmath-fp64.bc"
// SYCL_DEVICE_LIB_TSAN-SAME: "{{.*}}libsycl-imf.bc"
// SYCL_DEVICE_LIB_TSAN-SAME: "{{.*}}libsycl-imf-fp64.bc"
// SYCL_DEVICE_LIB_TSAN-SAME: "{{.*}}libsycl-imf-bf16.bc"
// SYCL_DEVICE_LIB_TSAN-SAME: "{{.*}}libsycl-fallback-cassert.bc"
// SYCL_DEVICE_LIB_TSAN-SAME: "{{.*}}libsycl-fallback-cstring.bc"
// SYCL_DEVICE_LIB_TSAN-SAME: "{{.*}}libsycl-fallback-complex.bc"
// SYCL_DEVICE_LIB_TSAN-SAME: "{{.*}}libsycl-fallback-complex-fp64.bc"
// SYCL_DEVICE_LIB_TSAN-SAME: "{{.*}}libsycl-fallback-cmath.bc"
// SYCL_DEVICE_LIB_TSAN-SAME: "{{.*}}libsycl-fallback-cmath-fp64.bc"
// SYCL_DEVICE_LIB_TSAN-SAME: "{{.*}}libsycl-fallback-imf.bc"
// SYCL_DEVICE_LIB_TSAN-SAME: "{{.*}}libsycl-fallback-imf-fp64.bc"
// SYCL_DEVICE_LIB_TSAN-SAME: "{{.*}}libsycl-fallback-imf-bf16.bc"
// SYCL_DEVICE_LIB_TSAN-SAME: "{{.*}}libsycl-tsan.bc"
// SYCL_DEVICE_TSAN_MACRO: "-cc1"
// SYCL_DEVICE_TSAN_MACRO-SAME: "USE_SYCL_DEVICE_TSAN"
// SYCL_DEVICE_TSAN_MACRO: llvm-link{{.*}} "-only-needed"
// SYCL_DEVICE_TSAN_MACRO-SAME: "{{.*}}libsycl-tsan.bc"

/// ###########################################################################
/// test behavior of linking libsycl-tsan-pvc for PVC target AOT compilation when tsan flag is applied.
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=thread -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_TSAN_PVC
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device pvc" --no-offload-new-driver %s \
// RUN: --sysroot=%S/Inputs/SYCL -Xarch_device -fsanitize=thread -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_TSAN_PVC
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device pvc" --no-offload-new-driver %s \
// RUN: --sysroot=%S/Inputs/SYCL -Xarch_device -fsanitize=thread -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_TSAN_PVC
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device 12.60.7" --no-offload-new-driver %s \
// RUN: --sysroot=%S/Inputs/SYCL -Xarch_device -fsanitize=thread -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_TSAN_PVC
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xs "-device 12.60.7" --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=thread -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_TSAN_PVC
// SYCL_DEVICE_LIB_TSAN_PVC: llvm-link{{.*}} "-only-needed" "{{.*}}libsycl-crt.bc"
// SYCL_DEVICE_LIB_TSAN_PVC-SAME: "{{.*}}libsycl-tsan-pvc.bc"


/// ###########################################################################
/// test behavior of linking libsycl-tsan-cpu for CPU target AOT compilation when tsan flag is applied.
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 --no-offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=thread -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_TSAN_CPU
// SYCL_DEVICE_LIB_TSAN_CPU: llvm-link{{.*}} "-only-needed" "{{.*}}libsycl-crt.bc"
// SYCL_DEVICE_LIB_TSAN_CPU-SAME: "{{.*}}libsycl-tsan-cpu.bc"
