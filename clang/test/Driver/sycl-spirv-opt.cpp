///
/// Tests for -Xspirv-translator
///

// RUN: %clangxx -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL -Xspirv-translator "foo" -### %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-SINGLE-TARGET

// RUN: %clangxx -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL -Xspirv-translator=spir64_gen "foo" -### %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-SINGLE-TARGET-UNUSED --implicit-check-not 'llvm-spirv{{.*}} "foo"'

// RUN: %clangxx -fsycl -fsycl-targets=spir64,spir64_gen --offload-new-driver --sysroot=%S/Inputs/SYCL -Xspirv-translator=spir64_gen "foo" -Xspirv-translator=spir64 "bar" -### %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-MULTIPLE-TARGET

// CHECK-SINGLE-TARGET: clang-linker-wrapper{{.*}} "--llvm-spirv-options=sycl:spir64-unknown-unknown=foo{{.*}}
// CHECK-SINGLE-TARGET-UNUSED: argument unused during compilation: '-Xspirv-translator=spir64_gen foo'
// CHECK-MULTIPLE-TARGET: clang-linker-wrapper{{.*}} "--llvm-spirv-options=sycl:spir64-unknown-unknown=bar"{{.*}}"--llvm-spirv-options=sycl:spir64_gen-unknown-unknown=foo"
