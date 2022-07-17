// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fsyntax-only -verify -flax-vector-conversions=none -Wconversion %s


typedef __attribute__(( ext_vector_type(2) )) float float2;
typedef double double2 __attribute__ ((ext_vector_type(2)));

static void splats(int i, long l, __uint128_t t, float f, double d) {
  float2 vf = f;
  double2 vd = d;
  vf = 2.0 + vf;
  vf = d + vf; // expected-warning {{implicit conversion loses floating-point precision}}
  vf = vf + 2.1; // expected-warning {{implicit conversion loses floating-point precision}}
}
