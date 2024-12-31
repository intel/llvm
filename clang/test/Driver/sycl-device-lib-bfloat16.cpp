///
/// Check if bfloat16 native and fallback libraries are linked correctly.
///

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
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc %s \
// RUN:   --sysroot=%S/Inputs/SYCL -### 2>&1 \
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


// Test test AOT-DG2 compilation uses native libs + native libs.
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_acm_g10 \
// RUN:   --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-NATIVE

// Test test AOT-PVC + AOT-DG2 compilation uses native libs + native libs.
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc,intel_gpu_acm_g10 \
// RUN:   --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-NATIVE-NATIVE

// Test test AOT-PVC + AOT-DG1 compilation uses native libs + native libs.
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_dg1,intel_gpu_acm_g10 \
// RUN:   --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK-FALLBACK


// Test test AOT-PVC + JIT compilation uses native libs + no libs
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc,spir64 \
// RUN:   --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-NATIVE-NONE
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc,spirv64 \
// RUN:   --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-NATIVE-NONE
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc,spir64-unknown-unknown \
// RUN:   --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-NATIVE-NONE
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc,spirv64-unknown-unknown \
// RUN:   --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-NATIVE-NONE

// Test test AOT-DG1 + JIT compilation uses native libs + no libs
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_dg1,spir64 \
// RUN:   --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK-NONE
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_dg1,spirv64 \
// RUN:   --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK-NONE
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_dg1,spir64-unknown-unknown \
// RUN:   --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK-NONE
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_dg1,spirv64-unknown-unknown \
// RUN:   --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK-NONE

// Test test AOT-PVC + JIT compilation + AOT-DG2 uses native libs + no libs + native libs
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc,spir64,intel_gpu_acm_g10 \
// RUN:   --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-NATIVE-NONE-NATIVE
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc,spirv64,intel_gpu_acm_g10 \
// RUN:   --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-NATIVE-NONE-NATIVE
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc,spir64-unknown-unknown,intel_gpu_acm_g10 \
// RUN:   --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-NATIVE-NONE-NATIVE
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc,spirv64-unknown-unknown,intel_gpu_acm_g10 \
// RUN:   --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-NATIVE-NONE-NATIVE

// Test test AOT-PVC + JIT compilation + AOT-DG1 uses fallback libs + no libs + fallback libs
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc,spir64,intel_gpu_dg1 \
// RUN:   --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK-NONE-FALLBACK
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc,spirv64,intel_gpu_dg1 \
// RUN:   --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK-NONE-FALLBACK
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc,spir64-unknown-unknown,intel_gpu_dg1 \
// RUN:   --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK-NONE-FALLBACK
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc,spirv64-unknown-unknown,intel_gpu_dg1 \
// RUN:   --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK-NONE-FALLBACK

// BFLOAT16-NOT: llvm-link{{.*}} "{{.*}}libsycl-{{fallback|native}}-bfloat16.bc"

// BFLOAT16-NATIVE: llvm-link{{.*}} "{{.*}}libsycl-native-bfloat16.bc"

// BFLOAT16-FALLBACK: llvm-link{{.*}} "{{.*}}libsycl-fallback-bfloat16.bc"

// BFLOAT16-NONE-NATIVE-NOT: llvm-link{{.*}} "{{.*}}-bfloat16.bc"
// BFLOAT16-NONE-NATIVE: llvm-link{{.*}} "{{.*}}libsycl-native-bfloat16.bc"

// BFLOAT16-NONE-FALLBACK-NOT: llvm-link{{.*}} "{{.*}}-bfloat16.bc"
// BFLOAT16-NONE-FALLBACK: llvm-link{{.*}} "{{.*}}libsycl-fallback-bfloat16.bc"

// BFLOAT16-FALLBACK-NATIVE: llvm-link{{.*}} "{{.*}}libsycl-fallback-bfloat16.bc"
// BFLOAT16-FALLBACK-NATIVE: {{.*}}libsycl-native-bfloat16.bc"

// BFLOAT16-FALLBACK-FALLBACK: llvm-link{{.*}} "{{.*}}libsycl-fallback-bfloat16.bc"
// BFLOAT16-FALLBACK-FALLBACK: "{{.*}}libsycl-fallback-bfloat16.bc"

// BFLOAT16-NATIVE-NATIVE: llvm-link{{.*}} "{{.*}}libsycl-native-bfloat16.bc"
// BFLOAT16-NATIVE-NATIVE: llvm-link{{.*}} "{{.*}}libsycl-native-bfloat16.bc"

// BFLOAT16-NATIVE-NONE: llvm-link{{.*}} "{{.*}}libsycl-native-bfloat16.bc"
// BFLOAT16-NATIVE-NONE-NOT: llvm-link{{.*}} "{{.*}}-bfloat16.bc"

// BFLOAT16-FALLBACK-NONE: llvm-link{{.*}} "{{.*}}libsycl-fallback-bfloat16.bc"
// BFLOAT16-FALLBACK-NONE-NOT: llvm-link{{.*}} "{{.*}}-bfloat16.bc"

// BFLOAT16-NATIVE-NONE-NATIVE: llvm-link{{.*}} "{{.*}}libsycl-native-bfloat16.bc"
// BFLOAT16-NATIVE-NONE-NATIVE-NOT: llvm-link{{.*}} "{{.*}}-bfloat16.bc"
// BFLOAT16-NATIVE-NONE-NATIVE: llvm-link{{.*}} "{{.*}}libsycl-native-bfloat16.bc"

// BFLOAT16-FALLBACK-NONE-FALLBACK: llvm-link{{.*}} "{{.*}}libsycl-fallback-bfloat16.bc"
// BFLOAT16-FALLBACK-NONE-FALLBACK-NOT: llvm-link{{.*}} "{{.*}}-bfloat16.bc"
// BFLOAT16-FALLBACK-NONE-FALLBACK: llvm-link{{.*}} "{{.*}}libsycl-fallback-bfloat16.bc"
