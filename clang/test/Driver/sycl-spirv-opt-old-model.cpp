///
/// Tests for -Xspirv-translator
///

// RUN: %clangxx -fsycl --no-offload-new-driver -Xspirv-translator "foo" -### %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-SINGLE-TARGET

// RUN: %clangxx -fsycl --no-offload-new-driver -Xspirv-translator=spir64_gen "foo" -### %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-SINGLE-TARGET-UNUSED --implicit-check-not 'llvm-spirv{{.*}} "foo"'

// RUN: %clangxx -fsycl --no-offload-new-driver -fsycl-targets=spir64,spir64_gen -Xspirv-translator=spir64_gen "foo" -Xspirv-translator=spir64 "bar" -### %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-MULTIPLE-TARGET --implicit-check-not 'llvm-spirv{{.*}} "foo" "bar"'

// CHECK-SINGLE-TARGET: llvm-spirv{{.*}} "foo"

// CHECK-SINGLE-TARGET-UNUSED: argument unused during compilation: '-Xspirv-translator=spir64_gen foo'

// CHECK-MULTIPLE-TARGET: llvm-spirv{{.*}} "bar"
// CHECK-MULTIPLE-TARGET: clang-offload-wrapper{{.*}} "-target=spir64" "-kind=sycl"
// CHECK-MULTIPLE-TARGET: llvm-spirv{{.*}} "foo"
// CHECK-MULTIPLE-TARGET: clang-offload-wrapper{{.*}} "-target=spir64_gen" "-kind=sycl"
