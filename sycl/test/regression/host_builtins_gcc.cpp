// RUN: %clangxx -fsycl -std=c++17 %s -o %t_clang_host.out
// RUN: %clangxx -fsycl -std=c++17 -fsycl-host-compiler=g++ -fsycl-host-compiler-options="-std=c++17" %s -o %t_gcc_host.out
// expected-no-diagnostics
//
// REQUIRES: linux
//
// Tests that using gcc as the host compiler correctly compiles SYCL builtins on
// host.

#include <cmath>
#include <sycl/sycl.hpp>

#define TEST_CASE_1(OP, IN, OUT_TY)                                            \
  static_assert(std::is_same_v<OUT_TY, decltype(sycl::OP(IN))>,                \
                "Failed for " #OP "/" #IN "/" #OUT_TY);                        \
  sycl::OP(IN);

#define TEST_CASE_2(OP, IN1, IN2, OUT_TY)                                      \
  static_assert(std::is_same_v<OUT_TY, decltype(sycl::OP(IN1, IN2))>,          \
                "Failed for " #OP "/" #IN1 "/" #IN2 "/" #OUT_TY);              \
  sycl::OP(IN1, IN2);

#define TEST_CASE_3(OP, IN1, IN2, IN3, OUT_TY)                                 \
  static_assert(std::is_same_v<OUT_TY, decltype(sycl::OP(IN1, IN2, IN3))>,     \
                "Failed for " #OP "/" #IN1 "/" #IN2 "/" #IN3 "/" #OUT_TY);     \
  sycl::OP(IN1, IN2, IN3);

#define TEST_CASES_F1(OP)                                                      \
  {                                                                            \
    sycl::half h;                                                              \
    float f;                                                                   \
    double d;                                                                  \
    TEST_CASE_1(OP, h, sycl::half)                                             \
    TEST_CASE_1(OP, f, float)                                                  \
    TEST_CASE_1(OP, d, double)                                                 \
  }

#define TEST_CASES_F2(OP)                                                      \
  {                                                                            \
    sycl::half h;                                                              \
    float f;                                                                   \
    double d;                                                                  \
    TEST_CASE_2(OP, h, h, sycl::half)                                          \
    TEST_CASE_2(OP, f, f, float)                                               \
    TEST_CASE_2(OP, d, d, double)                                              \
  }

#define TEST_CASES_F1IP1(OP)                                                   \
  {                                                                            \
    sycl::half h;                                                              \
    float f;                                                                   \
    double d;                                                                  \
    int *ip = nullptr;                                                         \
    TEST_CASE_2(OP, h, sycl::private_ptr<int>{ip}, sycl::half)                 \
    TEST_CASE_2(OP, f, sycl::private_ptr<int>{ip}, float)                      \
    TEST_CASE_2(OP, d, sycl::private_ptr<int>{ip}, double)                     \
  }

#define TEST_CASES_F1FP1(OP)                                                   \
  {                                                                            \
    sycl::half h;                                                              \
    float f;                                                                   \
    double d;                                                                  \
    TEST_CASE_2(OP, h, sycl::private_ptr<sycl::half>{&h}, sycl::half)          \
    TEST_CASE_2(OP, f, sycl::private_ptr<float>{&f}, float)                    \
    TEST_CASE_2(OP, d, sycl::private_ptr<double>{&d}, double)                  \
  }

#define TEST_CASES_F2IP1(OP)                                                   \
  {                                                                            \
    sycl::half h;                                                              \
    float f;                                                                   \
    double d;                                                                  \
    int *ip = nullptr;                                                         \
    TEST_CASE_3(OP, h, h, sycl::private_ptr<int>{ip}, sycl::half)              \
    TEST_CASE_3(OP, f, f, sycl::private_ptr<int>{ip}, float)                   \
    TEST_CASE_3(OP, d, d, sycl::private_ptr<int>{ip}, double)                  \
  }

int main() {
  TEST_CASES_F1(cos)
  TEST_CASES_F1(sin)
  TEST_CASES_F1(tan)
  TEST_CASES_F1(acos)
  TEST_CASES_F1(asin)
  TEST_CASES_F1(atan)
  TEST_CASES_F2(atan2)
  TEST_CASES_F1(cosh)
  TEST_CASES_F1(sinh)
  TEST_CASES_F1(tanh)
  TEST_CASES_F1(acosh)
  TEST_CASES_F1(asinh)
  TEST_CASES_F1(atanh)
  TEST_CASES_F1(exp)
  TEST_CASES_F1(exp2)
  TEST_CASES_F1IP1(frexp)
  TEST_CASES_F1(expm1)
  TEST_CASES_F1(log)
  TEST_CASES_F1(log2)
  TEST_CASES_F1(log10)
  TEST_CASES_F1FP1(modf)
  TEST_CASES_F2(pow)
  TEST_CASES_F1(sqrt)
  TEST_CASES_F1(cbrt)
  TEST_CASES_F2(hypot)
  TEST_CASES_F1(erf)
  TEST_CASES_F1(erfc)
  TEST_CASES_F1(tgamma)
  TEST_CASES_F1(lgamma)
  TEST_CASES_F1(ceil)
  TEST_CASES_F1(floor)
  TEST_CASES_F2(fmod)
  TEST_CASES_F1(trunc)
  TEST_CASES_F1(round)
  TEST_CASES_F1(rint)
  TEST_CASES_F2(remainder)
  TEST_CASES_F2IP1(remquo)
  TEST_CASES_F2(copysign)
  TEST_CASES_F2(nextafter)
  TEST_CASES_F2(fdim)
  TEST_CASES_F2(fmax)
  TEST_CASES_F2(fmin)

  unsigned int i;
  unsigned long l;
  TEST_CASE_1(nan, i, float)
  TEST_CASE_1(nan, l, double)

  return 0;
}
