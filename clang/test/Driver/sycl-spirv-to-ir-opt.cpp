///
/// Tests for -Xspirv-to-ir-wrapper
///

// RUN: %clangxx -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL -Xspirv-to-ir-wrapper "foo" -### %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-SINGLE-TARGET

// RUN: %clangxx -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL -Xspirv-to-ir-wrapper=spir64_gen "foo" -### %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-SINGLE-TARGET-UNUSED --implicit-check-not 'spirv-to-ir-wrapper-options{{.*}}=foo'

// RUN: %clangxx -fsycl -fsycl-targets=spir64,spir64_gen --offload-new-driver --sysroot=%S/Inputs/SYCL -Xspirv-to-ir-wrapper=spir64_gen "foo" -Xspirv-to-ir-wrapper=spir64 "bar" -### %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-MULTIPLE-TARGET

// Each token is forwarded as its own --spirv-to-ir-wrapper-options occurrence.
// RUN: %clangxx -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL -Xspirv-to-ir-wrapper "foo bar" -### %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-MULTIPLE-TOKENS

// RUN: %clangxx -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL -Xspirv-to-ir-wrapper '"foo bar"' -### %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-SPACE

// CHECK-SINGLE-TARGET: clang-linker-wrapper{{.*}} "--spirv-to-ir-wrapper-options=sycl:spir64-unknown-unknown=foo{{.*}}
// CHECK-SINGLE-TARGET-UNUSED: argument unused during compilation: '-Xspirv-to-ir-wrapper=spir64_gen foo'
// CHECK-MULTIPLE-TARGET: clang-linker-wrapper{{.*}} "--spirv-to-ir-wrapper-options=sycl:spir64-unknown-unknown=bar"{{.*}}"--spirv-to-ir-wrapper-options=sycl:spir64_gen-unknown-unknown=foo"
// CHECK-MULTIPLE-TOKENS: clang-linker-wrapper{{.*}} "--spirv-to-ir-wrapper-options=sycl:spir64-unknown-unknown=foo"{{.*}}"--spirv-to-ir-wrapper-options=sycl:spir64-unknown-unknown=bar"
// CHECK-SPACE: clang-linker-wrapper{{.*}} "--spirv-to-ir-wrapper-options=sycl:spir64-unknown-unknown=foo bar"
