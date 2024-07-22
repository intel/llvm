///
/// Tests for -Xspirv-to-ir-wrapper
///

// RUN: touch %tfoo.o
// RUN: %clangxx -fsycl --no-offload-new-driver -Xspirv-to-ir-wrapper "foo" -### %tfoo.o 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-SINGLE-TARGET

// RUN: %clangxx -fsycl --no-offload-new-driver -Xspirv-to-ir-wrapper=spir64_gen "foo" -### %tfoo.o 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-SINGLE-TARGET-UNUSED --implicit-check-not 'spirv-to-ir-wrapper{{.*}} "foo"'

// RUN: %clangxx -fsycl --no-offload-new-driver -fsycl-targets=spir64,spir64_gen -Xspirv-to-ir-wrapper=spir64_gen "foo" -Xspirv-to-ir-wrapper=spir64 "bar" -### %tfoo.o 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-MULTIPLE-TARGET --implicit-check-not 'spirv-to-ir-wrapper{{.*}} "foo" "bar"'

// CHECK-SINGLE-TARGET: spirv-to-ir-wrapper{{.*}} "foo"

// CHECK-SINGLE-TARGET-UNUSED: argument unused during compilation: '-Xspirv-to-ir-wrapper=spir64_gen foo'

// CHECK-MULTIPLE-TARGET: spirv-to-ir-wrapper{{.*}} "bar"
// CHECK-MULTIPLE-TARGET: clang-offload-wrapper{{.*}} "-target=spir64" "-kind=sycl"
// CHECK-MULTIPLE-TARGET: spirv-to-ir-wrapper{{.*}} "foo"
// CHECK-MULTIPLE-TARGET: clang-offload-wrapper{{.*}} "-target=spir64_gen" "-kind=sycl"
