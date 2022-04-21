// RUN: %clang_cc1 %s -E -dM | FileCheck %s
// RUN: %clang_cc1 %s  -fdeclare-spirv-builtins  -E -dM | FileCheck --check-prefix=CHECK-SPIRV %s

// CHECK-NOT:#define __SPIRV_BUILTIN_DECLARATIONS__

// CHECK-SPIRV:#define __SPIRV_BUILTIN_DECLARATIONS__
