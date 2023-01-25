///
/// Check if bfloat16 native and fallback libraries are added on Windows
///

// REQUIRES: windows
// UNSUPPORTED: cuda

/// ###########################################################################
/// Test that no bfloat16 libraries are added in JIT mode.
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16

// Test that fallback bfloat16 libraries are added in JIT mode with
// generic target.
// RUN: %clangxx -fsycl -fsycl-targets=spir64 %s \
// RUN:   --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK

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

// Test that a mixed JIT + AOT-PVC  compilation uses no libs + fallback libs.
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

// BFLOAT16: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-crt.obj" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex.obj" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.obj" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath.obj" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.obj" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-mscv-math.obj" "-output={{.*}}libsycl-msvc-math-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf.obj" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.obj" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.obj" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.obj" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.obj" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.obj" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.obj" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.obj" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.obj" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.obj" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.obj" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-NOT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-{{fallback|native}}-bfloat16.obj" "-output={{.*}}libsycl-{{fallback|native}}-{{.*}}.o" "-unbundle"

// BFLOAT16-NATIVE: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-crt.obj" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex.obj" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.obj" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath.obj" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.obj" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-mscv-math.obj" "-output={{.*}}libsycl-msvc-math-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf.obj" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.obj" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.obj" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.obj" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.obj" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.obj" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.obj" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.obj" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.obj" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.obj" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.obj" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-native-bfloat16.obj"

// BFLOAT16-FALLBACK: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-crt.obj" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-complex.obj" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-complex-fp64.obj" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-cmath.obj" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.obj" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-mscv-math.obj" "-output={{.*}}libsycl-msvc-math-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-imf.obj" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-imf-fp64.obj" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-imf-bf16.obj" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.obj" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.obj" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-complex.obj" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.obj" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.obj" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-imf.obj" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.obj" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.obj" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-bfloat16.obj"

// BFLOAT16-NONE-NATIVE: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-crt.obj" "-output={{.*}}libsycl-crt-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex.obj" "-output={{.*}}libsycl-complex-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.obj" "-output={{.*}}libsycl-complex-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath.obj" "-output={{.*}}libsycl-cmath-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.obj" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-mscv-math.obj" "-output={{.*}}libsycl-mscv-math-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf.obj" "-output={{.*}}libsycl-imf-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.obj" "-output={{.*}}libsycl-imf-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.obj" "-output={{.*}}libsycl-imf-bf16-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.obj" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.obj" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.obj" "-output={{.*}}libsycl-fallback-complex-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.obj" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.obj" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.obj" "-output={{.*}}libsycl-fallback-imf-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.obj" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.obj" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-user-wrappers.obj" "-output={{.*}}libsycl-itt-user-wrappers-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-compiler-wrappers.obj" "-output={{.*}}libsycl-itt-compiler-wrappers-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-stubs.obj" "-output={{.*}}libsycl-itt-stubs-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: llvm-link{{.*}}
// BFLOAT16-NONE-NATIVE-NEXT: sycl-post-link{{.*}}
// BFLOAT16-NONE-NATIVE-NEXT: file-table-tform{{.*}}
// BFLOAT16-NONE-NATIVE-NEXT: llvm-foreach{{.*}}
// BFLOAT16-NONE-NATIVE-NEXT: file-table-tform{{.*}}
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-wrapper{{.*}}
// BFLOAT16-NONE-NATIVE-NEXT: llc{{.*}}
// BFLOAT16-NONE-NATIVE-NEXT: clang-16{{.*}}
// BFLOAT16-NONE-NATIVE-NEXT: llvm-link{{.*}}
// BFLOAT16-NONE-NATIVE: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-crt.obj" "-output={{.*}}libsycl-crt-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex.obj" "-output={{.*}}libsycl-complex-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.obj" "-output={{.*}}libsycl-complex-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath.obj" "-output={{.*}}libsycl-cmath-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.obj" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-mscv-math.obj" "-output={{.*}}libsycl-mscv-math-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf.obj" "-output={{.*}}libsycl-imf-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.obj" "-output={{.*}}libsycl-imf-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.obj" "-output={{.*}}libsycl-imf-bf16-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.obj" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.obj" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.obj" "-output={{.*}}libsycl-fallback-complex-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.obj" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.obj" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.obj" "-output={{.*}}libsycl-fallback-imf-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.obj" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.obj" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-native-bfloat16.obj"

// BFLOAT16-NONE-FALLBACK: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-crt.obj" "-output={{.*}}libsycl-crt-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex.obj" "-output={{.*}}libsycl-complex-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.obj" "-output={{.*}}libsycl-complex-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath.obj" "-output={{.*}}libsycl-cmath-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.obj" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-mscv-math.obj" "-output={{.*}}libsycl-mscv-math-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf.obj" "-output={{.*}}libsycl-imf-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.obj" "-output={{.*}}libsycl-imf-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.obj" "-output={{.*}}libsycl-imf-bf16-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.obj" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.obj" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.obj" "-output={{.*}}libsycl-fallback-complex-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.obj" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.obj" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.obj" "-output={{.*}}libsycl-fallback-imf-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.obj" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.obj" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-user-wrappers.obj" "-output={{.*}}libsycl-itt-user-wrappers-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-compiler-wrappers.obj" "-output={{.*}}libsycl-itt-compiler-wrappers-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-stubs.obj" "-output={{.*}}libsycl-itt-stubs-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: llvm-link{{.*}}
// BFLOAT16-NONE-FALLBACK-NEXT: sycl-post-link{{.*}}
// BFLOAT16-NONE-FALLBACK-NEXT: file-table-tform{{.*}}
// BFLOAT16-NONE-FALLBACK-NEXT: llvm-foreach{{.*}}
// BFLOAT16-NONE-FALLBACK-NEXT: file-table-tform{{.*}}
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-wrapper{{.*}}
// BFLOAT16-NONE-FALLBACK-NEXT: llc{{.*}}
// BFLOAT16-NONE-FALLBACK-NEXT: clang-16{{.*}}
// BFLOAT16-NONE-FALLBACK-NEXT: llvm-link{{.*}}
// BFLOAT16-NONE-FALLBACK: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-crt.obj" "-output={{.*}}libsycl-crt-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex.obj" "-output={{.*}}libsycl-complex-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.obj" "-output={{.*}}libsycl-complex-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath.obj" "-output={{.*}}libsycl-cmath-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.obj" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-mscv-math.obj" "-output={{.*}}libsycl-mscv-math-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf.obj" "-output={{.*}}libsycl-imf-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.obj" "-output={{.*}}libsycl-imf-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.obj" "-output={{.*}}libsycl-imf-bf16-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.obj" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.obj" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.obj" "-output={{.*}}libsycl-fallback-complex-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.obj" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.obj" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.obj" "-output={{.*}}libsycl-fallback-imf-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.obj" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.obj" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.obj" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-bfloat16.obj"

// BFLOAT16-FALLBACK-NATIVE: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-crt.obj" "-output={{.*}}libsycl-crt-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-complex.obj" "-output={{.*}}libsycl-complex-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.obj" "-output={{.*}}libsycl-complex-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-cmath.obj" "-output={{.*}}libsycl-cmath-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.obj" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-mscv-math.obj" "-output={{.*}}libsycl-mscv-math-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-imf.obj" "-output={{.*}}libsycl-imf-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.obj" "-output={{.*}}libsycl-imf-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.obj" "-output={{.*}}libsycl-imf-bf16-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.obj" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.obj" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.obj" "-output={{.*}}libsycl-fallback-complex-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.obj" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.obj" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.obj" "-output={{.*}}libsycl-fallback-imf-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.obj" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.obj" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-bfloat16.obj" "-output={{.*}}libsycl-fallback-bfloat16-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-itt-user-wrappers.obj" "-output={{.*}}libsycl-itt-user-wrappers-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-itt-compiler-wrappers.obj" "-output={{.*}}libsycl-itt-compiler-wrappers-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-itt-stubs.obj" "-output={{.*}}libsycl-itt-stubs-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: llvm-link{{.*}}
// BFLOAT16-FALLBACK-NATIVE-NEXT: sycl-post-link{{.*}}
// BFLOAT16-FALLBACK-NATIVE-NEXT: file-table-tform{{.*}}
// BFLOAT16-FALLBACK-NATIVE-NEXT: llvm-foreach{{.*}}
// BFLOAT16-FALLBACK-NATIVE-NEXT: llvm-foreach{{.*}}
// BFLOAT16-FALLBACK-NATIVE-NEXT: file-table-tform{{.*}}
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-wrapper{{.*}}
// BFLOAT16-FALLBACK-NATIVE-NEXT: llc{{.*}}
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-16{{.*}}
// BFLOAT16-FALLBACK-NATIVE-NEXT: llvm-link{{.*}}
// BFLOAT16-FALLBACK-NATIVE: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-crt.obj" "-output={{.*}}libsycl-crt-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex.obj" "-output={{.*}}libsycl-complex-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.obj" "-output={{.*}}libsycl-complex-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath.obj" "-output={{.*}}libsycl-cmath-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.obj" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-mscv-math.obj" "-output={{.*}}libsycl-mscv-math-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf.obj" "-output={{.*}}libsycl-imf-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.obj" "-output={{.*}}libsycl-imf-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.obj" "-output={{.*}}libsycl-imf-bf16-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.obj" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.obj" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.obj" "-output={{.*}}libsycl-fallback-complex-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.obj" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.obj" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.obj" "-output={{.*}}libsycl-fallback-imf-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.obj" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.obj" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-native-bfloat16.obj"

// BFLOAT16-FALLBACK-FALLBACK: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-crt.obj" "-output={{.*}}libsycl-crt-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-complex.obj" "-output={{.*}}libsycl-complex-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.obj" "-output={{.*}}libsycl-complex-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-cmath.obj" "-output={{.*}}libsycl-cmath-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.obj" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-mscv-math.obj" "-output={{.*}}libsycl-mscv-math-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-imf.obj" "-output={{.*}}libsycl-imf-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.obj" "-output={{.*}}libsycl-imf-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.obj" "-output={{.*}}libsycl-imf-bf16-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.obj" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.obj" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.obj" "-output={{.*}}libsycl-fallback-complex-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.obj" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.obj" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.obj" "-output={{.*}}libsycl-fallback-imf-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.obj" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.obj" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-bfloat16.obj" "-output={{.*}}libsycl-fallback-bfloat16-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-itt-user-wrappers.obj" "-output={{.*}}libsycl-itt-user-wrappers-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-itt-compiler-wrappers.obj" "-output={{.*}}libsycl-itt-compiler-wrappers-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-itt-stubs.obj" "-output={{.*}}libsycl-itt-stubs-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: llvm-link{{.*}}
// BFLOAT16-FALLBACK-FALLBACK-NEXT: sycl-post-link{{.*}}
// BFLOAT16-FALLBACK-FALLBACK-NEXT: file-table-tform{{.*}}
// BFLOAT16-FALLBACK-FALLBACK-NEXT: llvm-foreach{{.*}}
// BFLOAT16-FALLBACK-FALLBACK-NEXT: llvm-foreach{{.*}}
// BFLOAT16-FALLBACK-FALLBACK-NEXT: file-table-tform{{.*}}
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-wrapper{{.*}}
// BFLOAT16-FALLBACK-FALLBACK-NEXT: llc{{.*}}
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-16{{.*}}
// BFLOAT16-FALLBACK-FALLBACK-NEXT: llvm-link{{.*}}
// BFLOAT16-FALLBACK-FALLBACK: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-crt.obj" "-output={{.*}}libsycl-crt-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex.obj" "-output={{.*}}libsycl-complex-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.obj" "-output={{.*}}libsycl-complex-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath.obj" "-output={{.*}}libsycl-cmath-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.obj" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-mscv-math.obj" "-output={{.*}}libsycl-mscv-math-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf.obj" "-output={{.*}}libsycl-imf-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.obj" "-output={{.*}}libsycl-imf-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.obj" "-output={{.*}}libsycl-imf-bf16-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.obj" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.obj" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.obj" "-output={{.*}}libsycl-fallback-complex-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.obj" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.obj" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.obj" "-output={{.*}}libsycl-fallback-imf-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.obj" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.obj" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.obj" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-bfloat16.obj"
