///
/// Check if bfloat16 native and fallback libraries are added on Linux
///

// UNSUPPORTED: system-windows
// UNSUPPORTED: cuda

/// ###########################################################################
/// test that no bfloat16 libraries are added in JIT mode
// RUN: %clangxx -fsycl %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=BFLOAT16 --dump-input=always

// test that no bfloat16 libraries are added in JIT mode with generic target
// R UN: %clangxx -fsycl -fsycl-targets=spir64 %s -### 2>&1 \
// R UN:   | FileCheck %s -check-prefix=BFLOAT16

// test that a PVC AOT compilation uses the native library
// R UN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device pvc" %s -### 2>&1 \
// R UN:   | FileCheck %s -check-prefix=BFLOAT16-NATIVE

// test that unless all targets support bfloat16, AOT compilation uses the fallback library
// R UN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device pvc,gen9" %s -### 2>&1 \
// R UN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK

// test that when all targets support bfloat16, AOT compilation uses the native library
// R UN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device pvc-sdv,ats-m75" %s -### 2>&1 \
// R UN:   | FileCheck %s -check-prefix=BFLOAT16-NATIVE

// test that a gen9 AOT compilation uses the fallback library
// R UN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device gen9" %s -### 2>&1 \
// R UN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK

// test that a generic AOT compilation uses the fallback library
// R UN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device *" %s -### 2>&1 \
// R UN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK

// test that a mixed JIT + AOT-PVC compilation uses no libs + fallback libs
// R UN: %clangxx -fsycl -fsycl-targets=spir64,spir64_gen -Xsycl-target-backend=spir64_gen "-device pvc" %s -### 2>&1 \
// R UN:   | FileCheck %s -check-prefix=BFLOAT16-NONE-NATIVE

// test that a mixed JIT + AOT-Gen9 compilation uses no libs + native libs
// R UN: %clangxx -fsycl -fsycl-targets=spir64,spir64_gen -Xsycl-target-backend=spir64_gen "-device gen9" %s -### 2>&1 \
// R UN:   | FileCheck %s -check-prefix=BFLOAT16-NONE-FALLBACK

// test that an AOT-CPU + AOT-PVC compilation fallback + fallback libs
// R UN: %clangxx -fsycl -fsycl-targets=spir64_x86_64,spir64_gen -Xsycl-target-backend=spir64_gen "-device pvc" %s -### 2>&1 \
// R UN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK-NATIVE

// test that an AOT-CPU + AOT-Gen9 compilation uses fallback + native libs
// R UN: %clangxx -fsycl -fsycl-targets=spir64_x86_64,spir64_gen -Xsycl-target-backend=spir64_gen "-device gen9" %s -### 2>&1 \
// R UN:   | FileCheck %s -check-prefix=BFLOAT16-FALLBACK-FALLBACK

// BFLOAT16:      clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// B FLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// B FLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// B FLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// B FLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// B FLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// B FLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// B FLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// B FLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// B FLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// B FLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// B FLOAT16-NOT:  clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-{{fallback|native}}-bfloat16.o" "-output={{.*}}libsycl-{{fallback|native}}-{{.*}}.o" "-unbundle"

// B FLOAT16-NATIVE:      clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// B FLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// B FLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// B FLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// B FLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// B FLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// B FLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// B FLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// B FLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// B FLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// B FLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// B FLOAT16-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-native-bfloat16.o"

// B FLOAT16-FALLBACK:      clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-{{spir64_gen-|spir64-}}unknown-unknown" "-input={{.*}}libsycl-fallback-bfloat16.o"

// B FLOAT16-NONE-NATIVE:      clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-user-wrappers.o" "-output={{.*}}libsycl-itt-user-wrappers-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-compiler-wrappers.o" "-output={{.*}}libsycl-itt-compiler-wrappers-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-stubs.o" "-output={{.*}}libsycl-itt-stubs-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: llvm-link{{.*}}
// B FLOAT16-NONE-NATIVE-NEXT: sycl-post-link{{.*}}
// B FLOAT16-NONE-NATIVE-NEXT: file-table-tform{{.*}}
// B FLOAT16-NONE-NATIVE-NEXT: llvm-foreach{{.*}}
// B FLOAT16-NONE-NATIVE-NEXT: file-table-tform{{.*}}
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-wrapper{{.*}}
// B FLOAT16-NONE-NATIVE-NEXT: llc{{.*}}
// B FLOAT16-NONE-NATIVE-NEXT: clang-16{{.*}}
// B FLOAT16-NONE-NATIVE-NEXT: llvm-link{{.*}}
// B FLOAT16-NONE-NATIVE:      clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-native-bfloat16.o"

// B FLOAT16-NONE-FALLBACK:      clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-user-wrappers.o" "-output={{.*}}libsycl-itt-user-wrappers-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-compiler-wrappers.o" "-output={{.*}}libsycl-itt-compiler-wrappers-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown" "-input={{.*}}libsycl-itt-stubs.o" "-output={{.*}}libsycl-itt-stubs-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: llvm-link{{.*}}
// B FLOAT16-NONE-FALLBACK-NEXT: sycl-post-link{{.*}}
// B FLOAT16-NONE-FALLBACK-NEXT: file-table-tform{{.*}}
// B FLOAT16-NONE-FALLBACK-NEXT: llvm-foreach{{.*}}
// B FLOAT16-NONE-FALLBACK-NEXT: file-table-tform{{.*}}
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-wrapper{{.*}}
// B FLOAT16-NONE-FALLBACK-NEXT: llc{{.*}}
// B FLOAT16-NONE-FALLBACK-NEXT: clang-16{{.*}}
// B FLOAT16-NONE-FALLBACK-NEXT: llvm-link{{.*}}
// B FLOAT16-NONE-FALLBACK:      clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// B FLOAT16-NONE-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-bfloat16.o"

// B FLOAT16-FALLBACK-NATIVE:      clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-bfloat16.o" "-output={{.*}}libsycl-fallback-bfloat16-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-itt-user-wrappers.o" "-output={{.*}}libsycl-itt-user-wrappers-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-itt-compiler-wrappers.o" "-output={{.*}}libsycl-itt-compiler-wrappers-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-itt-stubs.o" "-output={{.*}}libsycl-itt-stubs-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: llvm-link{{.*}}
// B FLOAT16-FALLBACK-NATIVE-NEXT: sycl-post-link{{.*}}
// B FLOAT16-FALLBACK-NATIVE-NEXT: file-table-tform{{.*}}
// B FLOAT16-FALLBACK-NATIVE-NEXT: llvm-foreach{{.*}}
// B FLOAT16-FALLBACK-NATIVE-NEXT: llvm-foreach{{.*}}
// B FLOAT16-FALLBACK-NATIVE-NEXT: file-table-tform{{.*}}
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-wrapper{{.*}}
// B FLOAT16-FALLBACK-NATIVE-NEXT: llc{{.*}}
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-16{{.*}}
// B FLOAT16-FALLBACK-NATIVE-NEXT: llvm-link{{.*}}
// B FLOAT16-FALLBACK-NATIVE:      clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-NATIVE-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-native-bfloat16.o"

// B FLOAT16-FALLBACK-FALLBACK:      clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-fallback-bfloat16.o" "-output={{.*}}libsycl-fallback-bfloat16-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-itt-user-wrappers.o" "-output={{.*}}libsycl-itt-user-wrappers-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-itt-compiler-wrappers.o" "-output={{.*}}libsycl-itt-compiler-wrappers-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_x86_64-unknown-unknown" "-input={{.*}}libsycl-itt-stubs.o" "-output={{.*}}libsycl-itt-stubs-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: llvm-link{{.*}}
// B FLOAT16-FALLBACK-FALLBACK-NEXT: sycl-post-link{{.*}}
// B FLOAT16-FALLBACK-FALLBACK-NEXT: file-table-tform{{.*}}
// B FLOAT16-FALLBACK-FALLBACK-NEXT: llvm-foreach{{.*}}
// B FLOAT16-FALLBACK-FALLBACK-NEXT: llvm-foreach{{.*}}
// B FLOAT16-FALLBACK-FALLBACK-NEXT: file-table-tform{{.*}}
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-wrapper{{.*}}
// B FLOAT16-FALLBACK-FALLBACK-NEXT: llc{{.*}}
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-16{{.*}}
// B FLOAT16-FALLBACK-FALLBACK-NEXT: llvm-link{{.*}}
// B FLOAT16-FALLBACK-FALLBACK:      clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-crt.o" "-output={{.*}}libsycl-crt-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex.o" "-output={{.*}}libsycl-complex-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-complex-fp64.o" "-output={{.*}}libsycl-complex-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath.o" "-output={{.*}}libsycl-cmath-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-cmath-fp64.o" "-output={{.*}}libsycl-cmath-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf.o" "-output={{.*}}libsycl-imf-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-fp64.o" "-output={{.*}}libsycl-imf-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-imf-bf16.o" "-output={{.*}}libsycl-imf-bf16-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cassert.o" "-output={{.*}}libsycl-fallback-cassert-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cstring.o" "-output={{.*}}libsycl-fallback-cstring-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex.o" "-output={{.*}}libsycl-fallback-complex-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-complex-fp64.o" "-output={{.*}}libsycl-fallback-complex-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath.o" "-output={{.*}}libsycl-fallback-cmath-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-cmath-fp64.o" "-output={{.*}}libsycl-fallback-cmath-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf.o" "-output={{.*}}libsycl-fallback-imf-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-fp64.o" "-output={{.*}}libsycl-fallback-imf-fp64-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-imf-bf16.o" "-output={{.*}}libsycl-fallback-imf-bf16-{{.*}}.o" "-unbundle"
// B FLOAT16-FALLBACK-FALLBACK-NEXT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen-unknown-unknown" "-input={{.*}}libsycl-fallback-bfloat16.o"
