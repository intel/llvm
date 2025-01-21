// RUN: %clang_cc1 %s %s -std=c++17 -triple x86_64-linux-gnu -Wno-sycl-2020-compat -fsycl-is-device -verify -fsyntax-only -Wno-unused

// This would crash due to UniqueStableNameDiscriminator expecting only lambdas
// as unnamed records.
// expected-no-diagnostics

template <typename T1, typename T2>
class TC {};

template <typename T1, typename T2> void TFoo(T1 _T1, T2 _T2) {
 __builtin_sycl_unique_stable_name(TC<T1, T2>);
}
void use() {
  struct {float r1, r2, r12; } s;
  TFoo(1, s);
}
