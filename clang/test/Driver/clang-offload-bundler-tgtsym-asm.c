// REQUIRES: x86-registered-target

// This test check that clang-offload-bundler correctly handles embedded
//  bitcode objects with module-level inline assembly when generating
// .tgtsym section with defined symbols.

// RUN: %clang -target %itanium_abi_triple -c %s -o %t.o
// RUN: %clang -target x86_64-pc-linux-gnu -emit-llvm -c %s -o %t.tgt1

// RUN: clang-offload-bundler -type=o -targets=host-%itanium_abi_triple,openmp-x86_64-pc-linux-gnu -inputs=%t.o,%t.tgt1 -outputs=%t.fat.o
// RUN: llvm-readobj --string-dump=.tgtsym %t.fat.o | FileCheck %s

// CHECK: String dump of section '.tgtsym':
// CHECK: openmp-x86_64-pc-linux-gnu.foo

__asm__(".symver memcpy,memcpy@GLIBC_2.2.5");
void foo(void) {}
