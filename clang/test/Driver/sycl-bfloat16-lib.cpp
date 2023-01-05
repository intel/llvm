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

// BFLOAT16:      clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-NOT:  clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-{{fallback|native}}-bfloat16.o" "-output={{.*}}libsycl-{{fallback|native}}-{{.*}}.o" "-unbundle"

// BFLOAT16-NATIVE:      clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-native-bfloat16.o"

// BFLOAT16-FALLBACK:      clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-bfloat16.o"

// BFLOAT16-NONE-NATIVE:      clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-user-wrappers.o" "-output={{.*}}libsycl-itt-user-wrappers-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-compiler-wrappers.o" "-output={{.*}}libsycl-itt-compiler-wrappers-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-stubs.o" "-output={{.*}}libsycl-itt-stubs-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: llvm-link{{.*}}
// BFLOAT16-NONE-NATIVE-NEXT: sycl-post-link{{.*}}
// BFLOAT16-NONE-NATIVE-NEXT: file-table-tform{{.*}}
// BFLOAT16-NONE-NATIVE-NEXT: llvm-foreach{{.*}}
// BFLOAT16-NONE-NATIVE-NEXT: file-table-tform{{.*}}
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-wrapper{{.*}}
// BFLOAT16-NONE-NATIVE-NEXT: llc{{.*}}
// BFLOAT16-NONE-NATIVE-NEXT: clang-16{{.*}}
// BFLOAT16-NONE-NATIVE-NEXT: llvm-link{{.*}}
// BFLOAT16-NONE-NATIVE:      clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-native-bfloat16.o"

// BFLOAT16-NONE-FALLBACK:      clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-user-wrappers.o" "-output={{.*}}libsycl-itt-user-wrappers-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-compiler-wrappers.o" "-output={{.*}}libsycl-itt-compiler-wrappers-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-stubs.o" "-output={{.*}}libsycl-itt-stubs-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: llvm-link{{.*}}
// BFLOAT16-NONE-FALLBACK-NEXT: sycl-post-link{{.*}}
// BFLOAT16-NONE-FALLBACK-NEXT: file-table-tform{{.*}}
// BFLOAT16-NONE-FALLBACK-NEXT: llvm-foreach{{.*}}
// BFLOAT16-NONE-FALLBACK-NEXT: file-table-tform{{.*}}
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-wrapper{{.*}}
// BFLOAT16-NONE-FALLBACK-NEXT: llc{{.*}}
// BFLOAT16-NONE-FALLBACK-NEXT: clang-16{{.*}}
// BFLOAT16-NONE-FALLBACK-NEXT: llvm-link{{.*}}
// BFLOAT16-NONE-FALLBACK:      clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-bfloat16.o"

// BFLOAT16-FALLBACK-NATIVE:      clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-bfloat16.o" "-output={{.*}}libsycl-fallback-bfloat16-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-itt-user-wrappers.o" "-output={{.*}}libsycl-itt-user-wrappers-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-itt-compiler-wrappers.o" "-output={{.*}}libsycl-itt-compiler-wrappers-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-itt-stubs.o" "-output={{.*}}libsycl-itt-stubs-{{.*}}.o" "-unbundle"
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
// BFLOAT16-FALLBACK-NATIVE:      clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-native-bfloat16.o"

// BFLOAT16-FALLBACK-FALLBACK:      clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-bfloat16.o" "-output={{.*}}libsycl-fallback-bfloat16-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-itt-user-wrappers.o" "-output={{.*}}libsycl-itt-user-wrappers-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-itt-compiler-wrappers.o" "-output={{.*}}libsycl-itt-compiler-wrappers-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-itt-stubs.o" "-output={{.*}}libsycl-itt-stubs-{{.*}}.o" "-unbundle"
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
// BFLOAT16-FALLBACK-FALLBACK:      clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-bfloat16.o"
