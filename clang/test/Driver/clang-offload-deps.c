// REQUIRES: x86-registered-target

//
// Check help message.
//
// RUN: clang-offload-deps --help | FileCheck %s --check-prefix CHECK-HELP
// CHECK-HELP: {{.*}}OVERVIEW: A tool for creating dependence bitcode files for offload targets. Takes
// CHECK-HELP-NEXT: {{.*}}host image as input and produces bitcode files, one per offload target, with
// CHECK-HELP-NEXT: {{.*}}references to symbols that must be defined in target images.
// CHECK-HELP: {{.*}}USAGE: clang-offload-deps [options] <input file>
// CHECK-HELP: {{.*}}  --outputs=<string> - [<output file>,...]
// CHECK-HELP: {{.*}}  --targets=<string> - [<offload kind>-<target triple>,...]

//
// Create source image for reading dependencies from.
//
// RUN: %clang -target %itanium_abi_triple -c %s -o %t.host
// RUN: %clang -target x86_64-pc-linux-gnu -c %s -o %t.x86_64
// RUN: %clang -target spir64 -emit-llvm   -c %s -o %t.spir64
// RUN: clang-offload-bundler -type=o -targets=host-%itanium_abi_triple,openmp-x86_64-pc-linux-gnu,sycl-spir64 -inputs=%t.host,%t.x86_64,%t.spir64 -outputs=%t.fat

//
// Generate dependencies for targets and check contents of the output bitcode files.
//
// RUN: clang-offload-deps -targets=openmp-x86_64-pc-linux-gnu,sycl-spir64 -outputs=%t.deps.x86_64,%t.deps.spir64 %t.fat
// RUN: llvm-dis -o - %t.deps.x86_64 | FileCheck %s --check-prefixes=CHECK-DEPS-X86_64
// RUN: llvm-dis -o - %t.deps.spir64 | FileCheck %s --check-prefixes=CHECK-DEPS-SPIR64

// CHECK-DEPS-X86_64: target triple = "x86_64-pc-linux-gnu"
// CHECK-DEPS-X86_64: @bar = external global i8*
// CHECK-DEPS-X86_64: @foo = external global i8*
// CHECK-DEPS-X86_64: @offload.symbols = hidden local_unnamed_addr global [2 x i8*] [i8* bitcast (i8** @bar to i8*), i8* bitcast (i8** @foo to i8*)]

// CHECK-DEPS-SPIR64: target triple = "spir64"
// CHECK-DEPS-SPIR64: @bar = external global i8*
// CHECK-DEPS-SPIR64: @foo = external global i8*
// CHECK-DEPS-SPIR64: @llvm.used = appending global [2 x i8*] [i8* bitcast (i8** @bar to i8*), i8* bitcast (i8** @foo to i8*)], section "llvm.metadata"

//
// Check that input with no .tgtsym section is handled correctly.
//
// RUN: clang-offload-deps -targets=openmp-x86_64-pc-linux-gnu,sycl-spir64 -outputs=%t.empty.x86_64,%t.empty.spir64 %t.host
// RUN: llvm-dis -o - %t.empty.x86_64 | FileCheck %s --check-prefixes=CHECK-EMPTY-X86_64
// RUN: llvm-dis -o - %t.empty.spir64 | FileCheck %s --check-prefixes=CHECK-EMPTY-SPIR64

// CHECK-EMPTY-X86_64: target triple = "x86_64-pc-linux-gnu"
// CHECK-EMPTY-X86_64-NOT: @offload.symbols

// CHECK-EMPTY-SPIR64: target triple = "spir64"
// CHECK-EMPTY-SPIR64-NOT: @llvm.used

void foo(void) {}
void bar(void) {}
