///
/// Check if bfloat16 native and fallback libraries are added on Windows
///

// REQUIRES: windows

/// ###########################################################################
/// test that no bfloat16 libraries are added in JIT mode
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16

// test that fallback bfloat16 libraries are added in JIT mode with generic target
// RUN: %clangxx -fsycl -fsycl-targets=spir64 %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK

// test that a PVC AOT compilation uses the native library
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device pvc" %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-NATIVE

// test that a gen9 AOT compilation uses the fallback library
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device gen9" %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK

// test that a generic AOT compilation uses the fallback library
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device *" %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK

// BFLOAT16: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-crt.obj" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex.obj" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.obj" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath.obj" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.obj" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-mscv-math.obj" "-output={{.*}}libsycl-msvc-math-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf.obj" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.obj" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.obj" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.obj" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.obj" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.obj" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.obj" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.obj" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.obj" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NOT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-{{fallback|native}}-bfloat16.obj" "-output={{.*}}libsycl-{{fallback|native}}-{{.*}}.o" "-unbundle"

// BFLOAT16-NATIVE: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-crt.obj" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex.obj" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.obj" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath.obj" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.obj" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-mscv-math.obj" "-output={{.*}}libsycl-msvc-math-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf.obj" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.obj" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.obj" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.obj" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.obj" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.obj" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.obj" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.obj" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.obj" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-native-bfloat16.obj"

// BFLOAT16-FALLBACK: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-crt.obj" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-complex.obj" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-complex-fp64.obj" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-cmath.obj" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.obj" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-mscv-math.obj" "-output={{.*}}libsycl-msvc-math-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-imf.obj" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-imf-fp64.obj" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.obj" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.obj" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-complex.obj" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.obj" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.obj" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-imf.obj" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.obj" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-bfloat16.obj"
