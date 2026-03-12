///
/// Perform several driver tests for SYCL device libraries on Linux
///

// UNSUPPORTED: system-windows

/// ###########################################################################

/// test behavior of device library default link
// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_DEFAULT
// SYCL_DEVICE_LIB_LINK_DEFAULT: clang{{.*}} "-mlink-builtin-bitcode" "{{.*}}libsycl-crt.bc"
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-complex.bc
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-complex-fp64.bc
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-cmath.bc
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-cmath-fp64.bc
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-imf.bc
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-imf-fp64.bc
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-imf-bf16.bc
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-fallback-cstring.bc
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-fallback-complex.bc
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-fallback-complex-fp64.bc
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-fallback-cmath.bc
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-fallback-cmath-fp64.bc
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-fallback-imf.bc
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-fallback-imf-fp64.bc
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: {{.*}}libsycl-fallback-imf-bf16.bc
// SYCL_DEVICE_LIB_LINK_DEFAULT-SAME: "-mlink-builtin-bitcode-postopt"

/// ###########################################################################

/// test behavior of disabling all device libraries
// RUN: %clangxx -fsycl --offload-new-driver %s --no-offloadlib --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB
// SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB: {{.*}}clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown"
// SYCL_DEVICE_LIB_LINK_NO_DEVICE_LIB-NOT: libsycl-cmath.bc

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
// SYCL_DEVICE_LIB_ASAN: clang{{.*}} "-mlink-builtin-bitcode" "{{.*}}libsycl-crt.bc
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-complex.bc
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-complex-fp64.bc
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-cmath.bc
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-cmath-fp64.bc
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-imf.bc
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-imf-fp64.bc
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-imf-bf16.bc
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-fallback-cstring.bc
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-fallback-complex.bc
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-fallback-complex-fp64.bc
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-fallback-cmath.bc
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-fallback-cmath-fp64.bc
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-fallback-imf.bc
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-fallback-imf-fp64.bc
// SYCL_DEVICE_LIB_ASAN-SAME: {{.*}}libsycl-fallback-imf-bf16.bc
// SYCL_DEVICE_LIB_ASAN-SAME: "-mlink-bitcode-file" "{{.*}}libsycl-asan.bc"

// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device "-fsanitize=address -DUSE_SYCL_DEVICE_ASAN" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_ASAN_MACRO
// SYCL_DEVICE_ASAN_MACRO: clang{{.*}} "-mlink-bitcode-file" "{{.*}}libsycl-asan.bc"
// SYCL_DEVICE_ASAN_MACRO-SAME: "USE_SYCL_DEVICE_ASAN"

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
// SYCL_DEVICE_LIB_ASAN_PVC: clang{{.*}} "-mlink-builtin-bitcode" "{{.*}}libsycl-crt.bc"
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-complex.
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-complex-fp64.
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-cmath.bc
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-cmath-fp64.bc
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-imf.bc
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-imf-fp64.bc
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-imf-bf16.bc
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-fallback-cstring.bc
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-fallback-complex.bc
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-fallback-complex-fp64.bc
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-fallback-cmath.bc
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-fallback-cmath-fp64.bc
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-fallback-imf.bc
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-fallback-imf-fp64.bc
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: {{.*}}libsycl-fallback-imf-bf16.bc
// SYCL_DEVICE_LIB_ASAN_PVC-SAME: "-mlink-bitcode-file" "{{.*}}libsycl-asan-pvc.bc"

/// ###########################################################################
/// test behavior of linking libsycl-asan-cpu for CPU target AOT compilation when asan flag is applied.
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=address -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_CPU
// SYCL_DEVICE_LIB_ASAN_CPU: clang{{.*}} "-mlink-builtin-bitcode" "{{.*}}libsycl-crt.bc"
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-complex.bc
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-complex-fp64.bc
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-cmath.bc
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-cmath-fp64.bc
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-imf.bc
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-imf-fp64.bc
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-imf-bf16.bc
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-fallback-cstring.bc
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-fallback-complex.bc
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-fallback-complex-fp64.bc
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-fallback-cmath.bc
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-fallback-cmath-fp64.bc
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-fallback-imf.bc
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-fallback-imf-fp64.bc
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: {{.*}}libsycl-fallback-imf-bf16.bc
// SYCL_DEVICE_LIB_ASAN_CPU-SAME: "-mlink-bitcode-file" "{{.*}}libsycl-asan-cpu.bc"

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
// SYCL_DEVICE_LIB_ASAN_DG2: clang{{.*}} "-mlink-builtin-bitcode" "{{.*}}libsycl-crt.bc"
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-complex.bc
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-complex-fp64.bc
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-cmath.bc
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-cmath-fp64.bc
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-imf.bc
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-imf-fp64.bc
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-imf-bf16.bc
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-fallback-cstring.bc
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-fallback-complex.bc
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-fallback-complex-fp64.bc
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-fallback-cmath.bc
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-fallback-cmath-fp64.bc
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-fallback-imf.bc
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-fallback-imf-fp64.bc
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: {{.*}}libsycl-fallback-imf-bf16.bc
// SYCL_DEVICE_LIB_ASAN_DG2-SAME: "-mlink-bitcode-file" "{{.*}}libsycl-asan-dg2.bc"

/// ###########################################################################
/// test behavior of linking libsycl-asan for multiple targets AOT compilation
/// when asan flag is applied.
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=address -Xsycl-target-backend "-device pvc,dg2" -### 2>&1 \
// RUN: | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_MUL
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -Xarch_device -fsanitize=address -Xsycl-target-backend=spir64_gen "-device pvc,dg2" -### 2>&1 \
// RUN: | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_ASAN_MUL
// SYCL_DEVICE_LIB_ASAN_MUL: clang{{.*}} "-mlink-builtin-bitcode" "{{.*}}libsycl-crt.bc"
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-complex.bc
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-complex-fp64.bc
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-cmath.bc
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-cmath-fp64.bc
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-imf.bc
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-imf-fp64.bc
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-imf-bf16.bc
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-fallback-cstring.bc
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-fallback-complex.bc
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-fallback-complex-fp64.bc
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-fallback-cmath.bc
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-fallback-cmath-fp64.bc
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-fallback-imf.bc
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-fallback-imf-fp64.bc
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: {{.*}}libsycl-fallback-imf-bf16.bc
// SYCL_DEVICE_LIB_ASAN_MUL-SAME: "-mlink-bitcode-file" "{{.*}}libsycl-asan.bc"

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
// SYCL_DEVICE_LIB_MSAN: clang{{.*}} "-mlink-builtin-bitcode" "{{.*}}libsycl-crt.bc"
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-complex.bc
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-complex-fp64.bc
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-cmath.bc
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-cmath-fp64.bc
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-imf.bc
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-imf-fp64.bc
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-imf-bf16.bc
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-fallback-cstring.bc
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-fallback-complex.bc
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-fallback-complex-fp64.bc
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-fallback-cmath.bc
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-fallback-cmath-fp64.bc
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-fallback-imf.bc
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-fallback-imf-fp64.bc
// SYCL_DEVICE_LIB_MSAN-SAME: {{.*}}libsycl-fallback-imf-bf16.bc
// SYCL_DEVICE_LIB_MSAN-SAME: "-mlink-bitcode-file" "{{.*}}libsycl-msan.bc"

// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device "-fsanitize=memory -DUSE_SYCL_DEVICE_MSAN" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_MSAN_MACRO
// SYCL_DEVICE_MSAN_MACRO: clang{{.*}} "-mlink-bitcode-file" "{{.*}}libsycl-msan.bc"
// SYCL_DEVICE_MSAN_MACRO-SAME: "USE_SYCL_DEVICE_MSAN"

/// test behavior of msan libdevice linking when -fsanitize=memory is available for AOT targets
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -fsanitize=memory -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_MSAN_PVC
// SYCL_DEVICE_LIB_MSAN_PVC: clang{{.*}} "-mlink-bitcode-file" "{{.*}}libsycl-msan-pvc.bc"

/// test behavior of msan libdevice linking when -fsanitize=memory is available for AOT targets
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -fsanitize=memory -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_MSAN_CPU
// SYCL_DEVICE_LIB_MSAN_CPU: clang{{.*}} "-mlink-bitcode-file" "{{.*}}libsycl-msan-cpu.bc"

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
// SYCL_DEVICE_LIB_TSAN: clang{{.*}} "-mlink-builtin-bitcode" "{{.*}}libsycl-crt.bc"
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-complex.
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-complex-fp64.
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-cmath.bc
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-cmath-fp64.bc
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-imf.bc
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-imf-fp64.bc
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-imf-bf16.bc
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-fallback-cstring.bc
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-fallback-complex.bc
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-fallback-complex-fp64.bc
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-fallback-cmath.bc
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-fallback-cmath-fp64.bc
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-fallback-imf.bc
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-fallback-imf-fp64.bc
// SYCL_DEVICE_LIB_TSAN-SAME: {{.*}}libsycl-fallback-imf-bf16.bc
// SYCL_DEVICE_LIB_TSAN-SAME: "-mlink-bitcode-file" "{{.*}}libsycl-tsan.bc"

// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -Xarch_device "-fsanitize=thread -DUSE_SYCL_DEVICE_TSAN" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_TSAN_MACRO
// SYCL_DEVICE_TSAN_MACRO: clang{{.*}} "-mlink-bitcode-file" "{{.*}}libsycl-tsan.bc"
// SYCL_DEVICE_TSAN_MACRO-SAME: "USE_SYCL_DEVICE_TSAN"

/// test behavior of tsan libdevice linking when -fsanitize=thread is available for AOT targets
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -fsanitize=thread -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_TSAN_PVC
// SYCL_DEVICE_LIB_TSAN_PVC: clang{{.*}} "-mlink-bitcode-file" "{{.*}}libsycl-tsan-pvc.bc"

/// test behavior of tsan libdevice linking when -fsanitize=thread is available for AOT targets
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 --offload-new-driver %s --sysroot=%S/Inputs/SYCL \
// RUN: -fsanitize=thread -### 2>&1 | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_TSAN_CPU
// SYCL_DEVICE_LIB_TSAN_CPU: clang{{.*}} "-mlink-bitcode-file" "{{.*}}libsycl-tsan-cpu.bc"
