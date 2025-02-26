// This test verifies that the correct macros are predefined.
// This test verifies that __builtin_va_list is processed without errors.
// RUN: %clang_cc1 -fsycl-is-device -triple spir-unknown-unknown -aux-triple i386-pc-windows-msvc  -E -dM -o - | FileCheck -match-full-lines %s --check-prefix=CHECK-MS64
// RUN: %clang_cc1 -fsycl-is-device -triple spirv32-unknown-unknown -aux-triple i386-pc-windows-msvc -E -dM -o - | FileCheck -match-full-lines %s --check-prefix=CHECK-MS64
// CHECK-MS64: #define _M_IX86 600
void foo(__builtin_va_list bvl) {
  char * VaList = bvl;
}
