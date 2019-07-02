// RUN: %clang_cc1 -triple x86_64-windows-msvc -fsycl-is-device -fsyntax-only -verify %s
// expected-no-diagnostics

void foo(__builtin_va_list bvl) {
  char * VaList = bvl;
  static_assert(sizeof(wchar_t) == 2, "sizeof wchar is 2 on Windows");
}
