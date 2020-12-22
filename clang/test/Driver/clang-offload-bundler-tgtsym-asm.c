// REQUIRES: x86-registered-target
// UNSUPPORTED: system-windows

// This test check that clang-offload-bundler correctly handles embedded
//  bitcode objects with module-level inline assembly when generating
// .tgtsym section with defined symbols.

// RUN: %clang -target %itanium_abi_triple            -c %s -o %t.o
// RUN: %clang -target x86_64-pc-linux-gnu -emit-llvm -c %s -o %t.tgt1
// RUN: %clang -target spir64              -emit-llvm -c %s -o %t.tgt2

// RUN: clang-offload-bundler -type=o -targets=host-%itanium_abi_triple,openmp-x86_64-pc-linux-gnu,sycl-spir64 -inputs=%t.o,%t.tgt1,%t.tgt2 -outputs=%t.fat.o
// RUN: llvm-readobj --string-dump=.tgtsym %t.fat.o | FileCheck %s

// CHECK: String dump of section '.tgtsym':
// CHECK-DAG: openmp-x86_64-pc-linux-gnu.foo
// CHECK-DAG: sycl-spir64.foo

__asm__(".symver memcpy,memcpy@GLIBC_2.2.5");
void foo(void) {}
