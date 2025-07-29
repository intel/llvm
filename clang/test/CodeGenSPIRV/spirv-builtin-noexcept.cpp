// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fdeclare-spirv-builtins -fsyntax-only %s -verify
// expected-no-diagnostics

void acos(float val) {
  static_assert(noexcept(__spirv_ocl_acos(val)));
}

void isnan(float a) {
  static_assert(noexcept(__spirv_IsNan(a)));
}
