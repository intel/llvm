// REQUIRES: x86-registered-target

// This test check that clang-offload-bundler adds .tgtsym section to the output
// file when creating a fat object. This section contains names of the external
// symbols defined in the embdedded target objects with target prefixes.

// RUN: %clang -target %itanium_abi_triple -c %s -o %t.o
// RUN: %clang -target x86_64-pc-linux-gnu -c %s -o %t.tgt1
// RUN: %clang -target spir64 -emit-llvm   -c %s -o %t.tgt2

// RUN: clang-offload-bundler -type=o -targets=host-%itanium_abi_triple,openmp-x86_64-pc-linux-gnu,sycl-spir64 -input=%t.o -input=%t.tgt1 -input=%t.tgt2 -output=%t.fat.o
// RUN: llvm-readobj --string-dump=.tgtsym %t.fat.o | FileCheck %s

// CHECK: String dump of section '.tgtsym':
// CHECK-DAG: openmp-x86_64-pc-linux-gnu.foo
// CHECK-DAG: openmp-x86_64-pc-linux-gnu.bar
// CHECK-DAG: sycl-spir64.foo
// CHECK-DAG: sycl-spir64.bar
// CHECK-NOT: undefined_func
// CHECK-NOT: static_func
// CHECK-NOT: static_used
// CHECK-NOT: sycl-spir64.llvm.used
// CHECK-NOT: sycl-spir64.llvm.compiler.used
// CHECK-NOT: sycl-spir64.const_as

// RUN: clang-offload-bundler --add-target-symbols-to-bundled-object=false -type=o -targets=host-%itanium_abi_triple,openmp-x86_64-pc-linux-gnu,sycl-spir64 -input=%t.o -input=%t.tgt1 -input=%t.tgt2 -output=%t.fat.no.tgtsym.o
// RUN: llvm-readobj --string-dump=.tgtsym %t.fat.no.tgtsym.o | FileCheck %s --check-prefix CHECK-NO-TGTSYM

// CHECK-NO-TGTSYM-NOT: String dump of section '.tgtsym':

// Check that -target-symbols-only adds .tgtsym without the target bundles.
// RUN: clang-offload-bundler -target-symbols-only -type=o -targets=host-%itanium_abi_triple,sycl-spir64 -input=%t.o -input=%t.tgt2 -output=%t.symsonly.o
// RUN: llvm-readobj --string-dump=.tgtsym %t.symsonly.o | FileCheck %s --check-prefix CHECK-SYMS-ONLY
// RUN: llvm-readobj --sections %t.symsonly.o | FileCheck %s --check-prefix CHECK-SYMS-ONLY-SECTIONS

// CHECK-SYMS-ONLY: String dump of section '.tgtsym':
// CHECK-SYMS-ONLY-DAG: sycl-spir64.foo
// CHECK-SYMS-ONLY-DAG: sycl-spir64.bar
// CHECK-SYMS-ONLY-SECTIONS-NOT: __CLANG_OFFLOAD_BUNDLE__

// RUN: not clang-offload-bundler -target-symbols-only -type=bc -targets=host-%itanium_abi_triple,sycl-spir64 -input=%t.tgt2 -input=%t.tgt2 -output=%t.symsonly.bc 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-SYMS-ONLY-ERR
// RUN: not clang-offload-bundler -target-symbols-only --add-target-symbols-to-bundled-object=false -type=o -targets=host-%itanium_abi_triple,sycl-spir64 -input=%t.o -input=%t.tgt2 -output=%t.symsonly.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-SYMS-ONLY-ERR2

// CHECK-SYMS-ONLY-ERR: -target-symbols-only can only be used for bundling object files (-type=o)
// CHECK-SYMS-ONLY-ERR2: -target-symbols-only cannot be used with -add-target-symbols-to-bundled-object=false

const __attribute__((opencl_constant)) char const_as[] = "abc";

extern void my_printf(__attribute__((opencl_constant)) const char *fmt);

extern void undefined_func(void);

void foo(void) {
  // We aim to create a gep operator in LLVM IR to have a use of const_as
  my_printf(&const_as[1]);
  undefined_func();
}

static void static_func(void) __attribute__((noinline));
static void static_func(void) {}

void bar(void) {
  static_func();
}

static void static_used(void) __attribute__((used));
static void static_used() {}
