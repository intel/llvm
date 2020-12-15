///
/// Perform several driver tests for SYCL device libraries on Windows
///
// REQUIRES: clang-driver, windows

/// ###########################################################################

/// test behavior of device library default link
// RUN: %clangxx -fsycl %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-crt.obj" "-outputs={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-complex.obj" "-outputs={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-complex-fp64.obj" "-outputs={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-cmath.obj" "-outputs={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-cmath-fp64.obj" "-outputs={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-fallback-cassert.obj" "-outputs={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-fallback-complex.obj" "-outputs={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-fallback-complex-fp64.obj" "-outputs={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-fallback-cmath.obj" "-outputs={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// SYCL_DEVICE_LIB_UNBUNDLE_DEFAULT-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-fallback-cmath-fp64.obj" "-outputs={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"

/// ###########################################################################

/// test behavior of disabling all device libraries
// RUN: %clangxx -fsycl %s -fno-sycl-device-lib -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB
// SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB: {{.*}}clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown-sycldevice"
// SYCL_DEVICE_LIB_UNBUNDLE_NO_DEVICE_LIB-NEXT: {{.*}}llvm-link{{.*}} {{.*}} "--suppress-warnings"

/// ###########################################################################
/// test llvm-link behavior for linking device libraries
// RUN: %clangxx -fsycl %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_LLVM_LINK_DEVICE_LIB
// SYCL_LLVM_LINK_DEVICE_LIB: llvm-link{{.*}}  "{{.*}}.bc" "-o" "{{.*}}.bc" "--suppress-warnings"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-crt.obj" "-outputs={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-complex.obj" "-outputs={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-complex-fp64.obj" "-outputs={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-cmath.obj" "-outputs={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-cmath-fp64.obj" "-outputs={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-fallback-cassert.obj" "-outputs={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-fallback-complex.obj" "-outputs={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-fallback-complex-fp64.obj" "-outputs={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-fallback-cmath.obj" "-outputs={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}libsycl-fallback-cmath-fp64.obj" "-outputs={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// SYCL_LLVM_LINK_DEVICE_LIB-NEXT: llvm-link{{.*}} "-only-needed" "{{.*}}" "-o" "{{.*}}.bc" "--suppress-warnings"

/// ###########################################################################
/// test llvm-link behavior for fno-sycl-device-lib
// RUN: %clangxx -fsycl -fno-sycl-device-lib %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_LLVM_LINK_NO_DEVICE_LIB
// SYCL_LLVM_LINK_NO_DEVICE_LIB: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-device"
// SYCL_LLVM_LINK_NO_DEVICE_LIB-NOT: llvm-link{{.*}}  "-only-needed"
// SYCL_LLVM_LINK_NO_DEVICE_LIB: sycl-post-link{{.*}}  "-symbols" "-spec-const=rt" "-o" "{{.*}}.table" "{{.*}}.bc"
