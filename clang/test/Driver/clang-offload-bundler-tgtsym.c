// REQUIRES: x86-registered-target

// This test check that clang-offload-bundler adds .tgtsym section to the output
// file when creating a fat object. This section contains names of the external
// symbols defined in the embdedded target objects with target prefixes.

// RUN: %clang -target %itanium_abi_triple -c %s -o %t.o
// RUN: %clang -target x86_64-pc-linux-gnu -c %s -o %t.tgt1
// RUN: %clang -target spir64 -emit-llvm   -c %s -o %t.tgt2

// RUN: clang-offload-bundler -type=o -targets=host-%itanium_abi_triple,openmp-x86_64-pc-linux-gnu,sycl-spir64 -inputs=%t.o,%t.tgt1,%t.tgt2 -outputs=%t.fat.o
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
