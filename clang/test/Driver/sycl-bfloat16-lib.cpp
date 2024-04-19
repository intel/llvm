///
/// Check if bfloat16 native and fallback libraries are added on Linux
///

// UNSUPPORTED: system-windows
// UNSUPPORTED: cuda

/// ###########################################################################
/// Test that no bfloat16 libraries are added in JIT mode.
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16 --dump-input=always

// Test that no bfloat16 libraries are added in JIT mode with generic target.
// RUN: %clangxx -fsycl -fsycl-targets=spir64 %s \
// RUN:   --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16

// Test that a PVC AOT compilation uses the native library.
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend \
// RUN:   "-device pvc" %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-NATIVE

// Test that unless all targets support bfloat16, AOT compilation uses the
// fallback library.
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend \
// RUN:   "-device pvc,gen9" %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK

// Test that when all targets support bfloat16, AOT compilation uses the
// native library.
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend \
// RUN:   "-device pvc-sdv,ats-m75" %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-NATIVE

// Test that a gen9 AOT compilation uses the fallback library.
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend \
// RUN:   "-device gen9" %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK

// Test that a generic AOT compilation uses the fallback library.
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend \
// RUN:   "-device *" %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK

// Test that a mixed JIT + AOT-PVC compilation uses no libs + fallback libs.
// RUN: %clangxx -fsycl -fsycl-targets=spir64,spir64_gen \
// RUN:   -Xsycl-target-backend=spir64_gen "-device pvc" %s \
// RUN:   --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-NONE-NATIVE

// Test that a mixed JIT + AOT-Gen9 compilation uses no libs + native libs.
// RUN: %clangxx -fsycl -fsycl-targets=spir64,spir64_gen \
// RUN:   -Xsycl-target-backend=spir64_gen "-device gen9" %s \
// RUN:   --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-NONE-FALLBACK

// Test that an AOT-CPU + AOT-PVC compilation fallback + fallback libs.
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64,spir64_gen \
// RUN:   -Xsycl-target-backend=spir64_gen "-device pvc" %s \
// RUN:   --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK-NATIVE

// Test that an AOT-CPU + AOT-Gen9 compilation uses fallback + native libs.
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64,spir64_gen \
// RUN:   -Xsycl-target-backend=spir64_gen "-device gen9" %s \
// RUN:   --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK-FALLBACK

// BFLOAT16-NOT:  clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-{{fallback|native}}-bfloat16.o" "-output={{.*}}libsycl-{{fallback|native}}-{{.*}}.o" "-unbundle"

// BFLOAT16-NATIVE: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-native-bfloat16.o"

// BFLOAT16-FALLBACK: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-bfloat16.o"

// BFLOAT16-NONE-NATIVE-NOT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}-bfloat16.o"
// BFLOAT16-NONE-NATIVE: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-native-bfloat16.o"

// BFLOAT16-NONE-FALLBACK-NOT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}-bfloat16.o"
// BFLOAT16-NONE-FALLBACK: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-bfloat16.o"

// BFLOAT16-FALLBACK-NATIVE: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-bfloat16.o" "-output={{.*}}libsycl-fallback-bfloat16-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-native-bfloat16.o"

// BFLOAT16-FALLBACK-FALLBACK: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-bfloat16.o" "-output={{.*}}libsycl-fallback-bfloat16-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-bfloat16.o"
