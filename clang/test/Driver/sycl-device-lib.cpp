///
/// Perform several driver tests for SYCL device libraries on Linux
///

// UNSUPPORTED: system-windows

/// ###########################################################################

/// test behavior of device library default link
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
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
/// test behavior of libsycl-asan.o linking when -fsanitize=address is available
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -fsanitize=address -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xsycl-target-frontend -fsanitize=address -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xsycl-target-frontend=spir64 -fsanitize=address -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device -fsanitize=address -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device "-fsanitize=address -DUSE_SYCL_DEVICE_ASAN" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device "-fsanitize=address -DUSE_SYCL_DEVICE_ASAN" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_ASAN_MACRO
// SYCL_DEVICE_LIB_ASAN: clang-linker-wrapper{{.*}} "-sycl-device-libraries
// SYCL_DEVICE_LIB_ASAN: {{.*}}libsycl-crt.new.o
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-complex.
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-complex-fp64.
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-cmath.new.o
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-cmath-fp64.new.o
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-imf.new.o
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-imf-fp64.new.o
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-imf-bf16.new.o
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-fallback-cassert.new.o
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-fallback-cstring.new.o
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-fallback-complex.new.o
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-fallback-complex-fp64.new.o
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-fallback-cmath.new.o
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-fallback-cmath-fp64.new.o
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-fallback-imf.new.o
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-fallback-imf-fp64.new.o
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-fallback-imf-bf16.new.o
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-asan.new.o
// SYCL_DEVICE_ASAN_MACRO: "-cc1"
// SYCL_DEVICE_ASAN_MACRO-SAME: "USE_SYCL_DEVICE_ASAN"
// SYCL_DEVICE_ASAN_MACRO: libsycl-asan.new.o


/// ###########################################################################
/// test behavior of linking libsycl-asan-pvc for PVC target AOT compilation when asan flag is applied.
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=address -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_PVC
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=address -Xsycl-target-backend "-device pvc" -### 2>&1 \
// RUN: | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_PVC
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=address -Xsycl-target-backend=spir64_gen "-device pvc" -### 2>&1 \
// RUN: | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_PVC
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=address -Xsycl-target-backend "-device 12.60.7" -### 2>&1 \
// RUN: | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_PVC
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=address -Xs "-device pvc" -### 2>&1 \
// RUN: | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_PVC
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=address -Xs "-device 12.60.7" -### 2>&1 \
// RUN: | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_PVC
// SYCL_DEVICE_LIB_ASAN_PVC: clang-linker-wrapper{{.*}} "-sycl-device-libraries
// SYCL_DEVICE_LIB_ASAN_PVC: {{.*}}libsycl-crt.new.o
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-complex.
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-complex-fp64.
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-cmath.new.o
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-cmath-fp64.new.o
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-imf.new.o
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-imf-fp64.new.o
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-imf-bf16.new.o
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-fallback-cassert.new.o
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-fallback-cstring.new.o
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-fallback-complex.new.o
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-fallback-complex-fp64.new.o
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-fallback-cmath.new.o
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-fallback-cmath-fp64.new.o
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-fallback-imf.new.o
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-fallback-imf-fp64.new.o
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-fallback-imf-bf16.new.o
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-asan-pvc.new.o

/// ###########################################################################
/// test behavior of linking libsycl-asan-cpu for CPU target AOT compilation when asan flag is applied.
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=address -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_CPU
// SYCL_DEVICE_LIB_ASAN_CPU: clang-linker-wrapper{{.*}} "-sycl-device-libraries
// SYCL_DEVICE_LIB_ASAN_CPU: {{.*}}libsycl-crt.new.o
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-complex.
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-complex-fp64.
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-cmath.new.o
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-cmath-fp64.new.o
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-imf.new.o
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-imf-fp64.new.o
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-imf-bf16.new.o
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-fallback-cassert.new.o
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-fallback-cstring.new.o
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-fallback-complex.new.o
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-fallback-complex-fp64.new.o
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-fallback-cmath.new.o
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-fallback-cmath-fp64.new.o
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-fallback-imf.new.o
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-fallback-imf-fp64.new.o
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-fallback-imf-bf16.new.o
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-asan-cpu.new.o

/// ###########################################################################
/// test behavior of linking libsycl-asan-dg2 for DG2 target AOT compilation when asan flag is applied.
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_dg2_g10 --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=address -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_DG2
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=address -Xsycl-target-backend "-device dg2" -### 2>&1 \
// RUN: | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_DG2
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=address -Xsycl-target-backend=spir64_gen "-device dg2" -### 2>&1 \
// RUN: | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_DG2
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=address -Xs "-device dg2" -### 2>&1 \
// RUN: | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_DG2
// SYCL_DEVICE_LIB_ASAN_DG2: clang-linker-wrapper{{.*}} "-sycl-device-libraries
// SYCL_DEVICE_LIB_ASAN_DG2: {{.*}}libsycl-crt.new.o
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-complex.
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-complex-fp64.
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-cmath.new.o
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-cmath-fp64.new.o
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-imf.new.o
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-imf-fp64.new.o
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-imf-bf16.new.o
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-fallback-cassert.new.o
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-fallback-cstring.new.o
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-fallback-complex.new.o
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-fallback-complex-fp64.new.o
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-fallback-cmath.new.o
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-fallback-cmath-fp64.new.o
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-fallback-imf.new.o
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-fallback-imf-fp64.new.o
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-fallback-imf-bf16.new.o
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-asan-dg2.new.o

/// ###########################################################################
/// test behavior of linking libsycl-asan for multiple targets AOT compilation
/// when asan flag is applied.
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=address -Xsycl-target-backend "-device pvc,dg2" -### 2>&1 \
// RUN: | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_MUL
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=address -Xsycl-target-backend=spir64_gen "-device pvc,dg2" -### 2>&1 \
// RUN: | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_MUL
// SYCL_DEVICE_LIB_ASAN_MUL: clang-linker-wrapper{{.*}} "-sycl-device-libraries
// SYCL_DEVICE_LIB_ASAN_MUL: {{.*}}libsycl-crt.new.o
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-complex.
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-complex-fp64.
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-cmath.new.o
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-cmath-fp64.new.o
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-imf.new.o
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-imf-fp64.new.o
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-imf-bf16.new.o
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-fallback-cassert.new.o
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-fallback-cstring.new.o
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-fallback-complex.new.o
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-fallback-complex-fp64.new.o
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-fallback-cmath.new.o
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-fallback-cmath-fp64.new.o
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-fallback-imf.new.o
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-fallback-imf-fp64.new.o
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-fallback-imf-bf16.new.o
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-asan.new.o


/// ###########################################################################
/// test behavior of libsycl-msan.o linking when -fsanitize=memory is available
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -fsanitize=memory -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_MSAN
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xsycl-target-frontend -fsanitize=memory -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_MSAN
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xsycl-target-frontend=spir64 -fsanitize=memory -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_MSAN
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device -fsanitize=memory -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_MSAN
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device "-fsanitize=memory -DUSE_SYCL_DEVICE_MSAN" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_MSAN
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device "-fsanitize=memory -DUSE_SYCL_DEVICE_MSAN" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_MSAN_MACRO
// SYCL_DEVICE_LIB_MSAN: clang-linker-wrapper{{.*}} "-sycl-device-libraries
// SYCL_DEVICE_LIB_MSAN: {{.*}}libsycl-crt.new.o
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-complex.
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-complex-fp64.
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-cmath.new.o
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-cmath-fp64.new.o
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-imf.new.o
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-imf-fp64.new.o
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-imf-bf16.new.o
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-fallback-cassert.new.o
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-fallback-cstring.new.o
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-fallback-complex.new.o
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-fallback-complex-fp64.new.o
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-fallback-cmath.new.o
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-fallback-cmath-fp64.new.o
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-fallback-imf.new.o
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-fallback-imf-fp64.new.o
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-fallback-imf-bf16.new.o
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-msan.new.o
// SYCL_DEVICE_MSAN_MACRO: "-cc1"
// SYCL_DEVICE_MSAN_MACRO-SAME: "USE_SYCL_DEVICE_MSAN"
// SYCL_DEVICE_MSAN_MACRO: libsycl-msan.new.o

/// test behavior of msan libdevice linking when -fsanitize=memory is available for AOT targets
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -fsanitize=memory -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_MSAN_PVC
// SYCL_DEVICE_LIB_MSAN_PVC: clang-linker-wrapper{{.*}} "-sycl-device-libraries
// SYCL_DEVICE_LIB_MSAN_PVC-SAME: {{.*}}libsycl-msan-pvc.new.o

/// test behavior of msan libdevice linking when -fsanitize=memory is available for AOT targets
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -fsanitize=memory -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_MSAN_CPU
// SYCL_DEVICE_LIB_MSAN_CPU: clang-linker-wrapper{{.*}} "-sycl-device-libraries
// SYCL_DEVICE_LIB_MSAN_CPU-SAME: {{.*}}libsycl-msan-cpu.new.o

/// ###########################################################################
/// test behavior of libsycl-tsan.o linking when -fsanitize=thread is available
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -fsanitize=thread -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_TSAN
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xsycl-target-frontend -fsanitize=thread -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_TSAN
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xsycl-target-frontend=spir64 -fsanitize=thread -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_TSAN
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device -fsanitize=thread -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_TSAN
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device "-fsanitize=thread -DUSE_SYCL_DEVICE_TSAN" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_TSAN
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device "-fsanitize=thread -DUSE_SYCL_DEVICE_TSAN" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_TSAN_MACRO
// SYCL_DEVICE_LIB_TSAN: clang-linker-wrapper{{.*}} "-sycl-device-libraries
// SYCL_DEVICE_LIB_TSAN: {{.*}}libsycl-crt.new.o
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-complex.
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-complex-fp64.
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-cmath.new.o
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-cmath-fp64.new.o
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-imf.new.o
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-imf-fp64.new.o
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-imf-bf16.new.o
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-fallback-cassert.new.o
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-fallback-cstring.new.o
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-fallback-complex.new.o
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-fallback-complex-fp64.new.o
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-fallback-cmath.new.o
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-fallback-cmath-fp64.new.o
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-fallback-imf.new.o
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-fallback-imf-fp64.new.o
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-fallback-imf-bf16.new.o
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-tsan.new.o
// SYCL_DEVICE_TSAN_MACRO: "-cc1"
// SYCL_DEVICE_TSAN_MACRO-SAME: "USE_SYCL_DEVICE_TSAN"
// SYCL_DEVICE_TSAN_MACRO: libsycl-tsan.new.o

/// test behavior of tsan libdevice linking when -fsanitize=thread is available for AOT targets
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -fsanitize=thread -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_TSAN_PVC
// SYCL_DEVICE_LIB_TSAN_PVC: clang-linker-wrapper{{.*}} "-sycl-device-libraries
// SYCL_DEVICE_LIB_TSAN_PVC-SAME: {{.*}}libsycl-tsan-pvc.new.o

/// test behavior of tsan libdevice linking when -fsanitize=thread is available for AOT targets
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -fsanitize=thread -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_TSAN_CPU
// SYCL_DEVICE_LIB_TSAN_CPU: clang-linker-wrapper{{.*}} "-sycl-device-libraries
// SYCL_DEVICE_LIB_TSAN_CPU-SAME: {{.*}}libsycl-tsan-cpu.new.o
