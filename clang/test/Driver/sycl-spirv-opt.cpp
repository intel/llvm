///
/// Tests for -Xspirv-translator
///

// RUN: %clangxx -fsycl --offload-new-driver -Xspirv-translator "foo" -### %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-SINGLE-TARGET

// RUN: %clangxx -fsycl --offload-new-driver -Xspirv-translator=spir64_gen "foo" -### %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-SINGLE-TARGET-UNUSED --implicit-check-not 'llvm-spirv{{.*}} "foo"'

// CHECK-SINGLE-TARGET: clang-linker-wrapper{{.*}} "--llvm-spirv-options=foo{{.*}}
// CHECK-SINGLE-TARGET-UNUSED: argument unused during compilation: '-Xspirv-translator=spir64_gen foo'
