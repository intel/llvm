///
/// Check if bfloat16 native and fallback libraries are added on Linux
///

// UNSUPPORTED: system-windows

/// ###########################################################################
/// test that no bfloat16 libraries are added in JIT mode
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16

// test that no bfloat16 libraries are added in JIT mode with generic target
// RUN: %clangxx -fsycl -fsycl-targets=spir64 %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16

// test that a PVC AOT compilation uses the native library
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device pvc" %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-NATIVE

// test that a gen9 AOT compilation uses the fallback library
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device gen9" %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK

// test that a generic AOT compilation uses the fallback library
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device *" %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK

// BFLOAT16: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NOT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-{{fallback|native}}-bfloat16.o" "-output={{.*}}libsycl-{{fallback|native}}-{{.*}}.o" "-unbundle"

// BFLOAT16-NATIVE: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-native-bfloat16.o"

// BFLOAT16-FALLBACK: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// BFLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-bfloat16.o"
