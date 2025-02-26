// This test verifies that the correct macros are predefined.
// This test checks that sizes of different types are as expected.
// This test verifies that __builtin_va_list is processed without errors.
//
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-windows -aux-triple x86_64-pc-windows-msvc  -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsycl-is-device -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-msvc  -fsyntax-only -verify %s
// expected-no-diagnostics
// RUN: %clang_cc1 -fsycl-is-device -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-msvc  -E -dM -o - | FileCheck -match-full-lines %s --check-prefix=CHECK-MS64
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -aux-triple x86_64-pc-windows-msvc  -E -dM -o - | FileCheck -match-full-lines %s --check-prefix=CHECK-MS64
// CHECK-MS64: #define _M_AMD64 100
// CHECK-MS64: #define _M_X64 100
typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef __SIZE_TYPE__ size_t;
typedef __INT64_TYPE__ int64_t;
typedef __INTMAX_TYPE__ intmax_t;
void foo(__builtin_va_list bvl) {
  char * VaList = bvl;
  static_assert(sizeof(long) == 4, "sizeof long is 4 on Windows (x86_64)");
  static_assert(sizeof(double) == 8, "sizeof double is 8 on Windows (x86_64)");
  static_assert(sizeof(intmax_t) == 8, "sizeof intmax_t is 8 on Windows (x86_64)");
  static_assert(sizeof(int64_t) == 8, "sizeof int64_t is 8 on Windows (x86_64)");
  static_assert(sizeof(size_t) == 8, "sizeof size_t is 8 on Windows (x86_64)");
  static_assert(sizeof(ptrdiff_t) == 8, "sizeof ptrdiff_t is 8 on Windows (x86_64)");
  static_assert(sizeof(int *) == 8, "sizeof IntPtr is 8 on Windows (x86_64)");
  static_assert(sizeof(wchar_t) == 2, "sizeof wchar is 2 on Windows (x86_64)");
}
