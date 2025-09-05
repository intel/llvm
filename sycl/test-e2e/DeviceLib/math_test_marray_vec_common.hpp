#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>

#include <cmath>

using namespace sycl;

template <typename T1, typename T2> class TypeHelper;

template <typename T> bool checkEqual(vec<T, 3> A, size_t B) {
  T TB = B;
  return A.x() == TB && A.y() == TB && A.z() == TB;
}

template <typename T> bool checkEqual(vec<T, 4> A, size_t B) {
  T TB = B;
  return A.x() == TB && A.y() == TB && A.z() == TB && A.w() == TB;
}

template <typename T, size_t N> bool checkEqual(marray<T, N> A, size_t B) {
  for (int i = 0; i < N; i++) {
    if (A[i] != B) {
      return false;
    }
  }
  return true;
}

#define OPERATOR(NAME)                                                         \
  template <typename T>                                                        \
  void math_test_##NAME(queue &deviceQueue, T result, T input, size_t ref) {   \
    {                                                                          \
      buffer<T, 1> buffer1(&result, 1);                                        \
      buffer<T, 1> buffer2(&input, 1);                                         \
      deviceQueue.submit([&](handler &cgh) {                                   \
        accessor<T, 1, access::mode::write, target::device> res_access(        \
            buffer1, cgh);                                                     \
        accessor<T, 1, access::mode::read, target::device> input_access(       \
            buffer2, cgh);                                                     \
        cgh.single_task<TypeHelper<class a##NAME, T>>(                         \
            [=]() { res_access[0] = NAME(input_access[0]); });                 \
      });                                                                      \
    }                                                                          \
    assert(checkEqual(result, ref));                                           \
  }

OPERATOR(cos)
OPERATOR(cospi)
OPERATOR(sin)
OPERATOR(sinpi)
OPERATOR(cosh)
OPERATOR(sinh)
OPERATOR(tan)
OPERATOR(tanpi)
OPERATOR(atan)
OPERATOR(atanpi)
OPERATOR(tanh)
OPERATOR(acos)
OPERATOR(acospi)
OPERATOR(asin)
OPERATOR(asinpi)
OPERATOR(acosh)
OPERATOR(asinh)
OPERATOR(atanh)
OPERATOR(cbrt)
OPERATOR(ceil)
OPERATOR(exp)
OPERATOR(exp2)
OPERATOR(exp10)
OPERATOR(expm1)
OPERATOR(tgamma)
OPERATOR(lgamma)
OPERATOR(erf)
OPERATOR(erfc)
OPERATOR(log)
OPERATOR(log2)
OPERATOR(log10)
OPERATOR(log1p)
OPERATOR(logb)
OPERATOR(sqrt)
OPERATOR(rsqrt)
OPERATOR(rint)
OPERATOR(round)
OPERATOR(trunc)

#undef OPERATOR

#define OPERATOR_2(NAME)                                                       \
  template <typename T>                                                        \
  void math_test_2_##NAME(queue &deviceQueue, T result, T input1, T input2,    \
                          size_t ref) {                                        \
    {                                                                          \
      buffer<T, 1> buffer1(&result, 1);                                        \
      buffer<T, 1> buffer2(&input1, 1);                                        \
      buffer<T, 1> buffer3(&input2, 1);                                        \
      deviceQueue.submit([&](handler &cgh) {                                   \
        accessor<T, 1, access::mode::write, target::device> res_access(        \
            buffer1, cgh);                                                     \
        accessor<T, 1, access::mode::read, target::device> input1_access(      \
            buffer2, cgh);                                                     \
        accessor<T, 1, access::mode::read, target::device> input2_access(      \
            buffer3, cgh);                                                     \
        cgh.single_task<TypeHelper<class a##NAME, T>>([=]() {                  \
          res_access[0] = NAME(input1_access[0], input2_access[0]);            \
        });                                                                    \
      });                                                                      \
    }                                                                          \
    assert(checkEqual(result, ref));                                           \
  }

OPERATOR_2(pow)
OPERATOR_2(powr)
OPERATOR_2(atan2)
OPERATOR_2(atan2pi)
OPERATOR_2(copysign)
OPERATOR_2(fdim)
OPERATOR_2(fmin)
OPERATOR_2(fmax)
OPERATOR_2(fmod)
OPERATOR_2(hypot)
OPERATOR_2(maxmag)
OPERATOR_2(minmag)
OPERATOR_2(nextafter)
OPERATOR_2(remainder)

#undef OPERATOR_2

#define OPERATOR_3(NAME)                                                       \
  template <typename T>                                                        \
  void math_test_3_##NAME(queue &deviceQueue, T result, T input1, T input2,    \
                          T input3, size_t ref) {                              \
    {                                                                          \
      buffer<T, 1> buffer1(&result, 1);                                        \
      buffer<T, 1> buffer2(&input1, 1);                                        \
      buffer<T, 1> buffer3(&input2, 1);                                        \
      buffer<T, 1> buffer4(&input3, 1);                                        \
      deviceQueue.submit([&](handler &cgh) {                                   \
        accessor<T, 1, access::mode::write, target::device> res_access(        \
            buffer1, cgh);                                                     \
        accessor<T, 1, access::mode::read, target::device> input1_access(      \
            buffer2, cgh);                                                     \
        accessor<T, 1, access::mode::read, target::device> input2_access(      \
            buffer3, cgh);                                                     \
        accessor<T, 1, access::mode::read, target::device> input3_access(      \
            buffer4, cgh);                                                     \
        cgh.single_task<TypeHelper<class a##NAME, T>>([=]() {                  \
          res_access[0] =                                                      \
              NAME(input1_access[0], input2_access[0], input3_access[0]);      \
        });                                                                    \
      });                                                                      \
    }                                                                          \
    assert(checkEqual(result, ref));                                           \
  }

OPERATOR_3(mad)
OPERATOR_3(mix)
OPERATOR_3(fma)

#undef OPERATOR_3

template <typename T> void math_tests_4(queue &deviceQueue) {
  math_test_tanh(deviceQueue, T{-1, -1, -1, -1}, T{0, 0, 0, 0}, 0);
  math_test_cosh(deviceQueue, T{-1, -1, -1, -1}, T{0, 0, 0, 0}, 1);
  math_test_sinh(deviceQueue, T{-1, -1, -1, -1}, T{0, 0, 0, 0}, 0);
  math_test_acos(deviceQueue, T{-1, -1, -1, -1}, T{1, 1, 1, 1}, 0);
  math_test_acospi(deviceQueue, T{-1, -1, -1, -1}, T{1, 1, 1, 1}, 0);
  math_test_acosh(deviceQueue, T{-1, -1, -1, -1}, T{1, 1, 1, 1}, 0);
  math_test_asin(deviceQueue, T{-1, -1, -1, -1}, T{0, 0, 0, 0}, 0);
  math_test_asinpi(deviceQueue, T{-1, -1, -1, -1}, T{0, 0, 0, 0}, 0);
  math_test_asinh(deviceQueue, T{-1, -1, -1, -1}, T{0, 0, 0, 0}, 0);
  math_test_cbrt(deviceQueue, T{-1, -1, -1, -1}, T{1, 1, 1, 1}, 1);
  math_test_atan(deviceQueue, T{-1, -1, -1, -1}, T{0, 0, 0, 0}, 0);
  math_test_atanpi(deviceQueue, T{-1, -1, -1, -1}, T{0, 0, 0, 0}, 0);
  math_test_atanh(deviceQueue, T{-1, -1, -1, -1}, T{0, 0, 0, 0}, 0);
  math_test_exp(deviceQueue, T{-1, -1, -1, -1}, T{0, 0, 0, 0}, 1);
  math_test_exp2(deviceQueue, T{-1, -1, -1, -1}, T{2, 2, 2, 2}, 4);
  math_test_exp10(deviceQueue, T{-1, -1, -1, -1}, T{2, 2, 2, 2}, 100);
  math_test_expm1(deviceQueue, T{-1, -1, -1, -1}, T{0, 0, 0, 0}, 0);
  math_test_ceil(deviceQueue, T{-1, -1, -1, -1}, T{0.6, 0.6, 0.6, 0.6}, 1);
  math_test_tgamma(deviceQueue, T{-1, -1, -1, -1}, T{1, 1, 1, 1}, 1);
  math_test_lgamma(deviceQueue, T{-1, -1, -1, -1}, T{1, 1, 1, 1}, 0);
  math_test_erf(deviceQueue, T{-1, -1, -1, -1}, T{0, 0, 0, 0}, 0);
  math_test_erfc(deviceQueue, T{-1, -1, -1, -1}, T{0, 0, 0, 0}, 1);
  math_test_2_pow(deviceQueue, T{-1, -1, -1, -1}, T{2, 2, 2, 2}, T{2, 2, 2, 2},
                  4);
  math_test_2_powr(deviceQueue, T{-1, -1, -1, -1}, T{2, 2, 2, 2}, T{2, 2, 2, 2},
                   4);
  math_test_2_atan2(deviceQueue, T{-1, -1, -1, -1}, T{0, 0, 0, 0},
                    T{2, 2, 2, 2}, 0);
  math_test_2_atan2pi(deviceQueue, T{-1, -1, -1, -1}, T{0, 0, 0, 0},
                      T{2, 2, 2, 2}, 0);
  math_test_2_copysign(deviceQueue, T{-1, -1, -1, -1}, T{-3, -3, -3, -3},
                       T{2, 2, 2, 2}, 3);
  math_test_2_fmin(deviceQueue, T{-1, -1, -1, -1}, T{2, 2, 2, 2}, T{3, 3, 3, 3},
                   2);
  math_test_2_fmax(deviceQueue, T{-1, -1, -1, -1}, T{2, 2, 2, 2}, T{3, 3, 3, 3},
                   3);
  math_test_2_hypot(deviceQueue, T{-1, -1, -1, -1}, T{4, 4, 4, 4},
                    T{3, 3, 3, 3}, 5);
  math_test_2_maxmag(deviceQueue, T{-1, -1, -1, -1}, T{-2, -2, -2, -2},
                     T{3, 3, 3, 3}, 3);
  math_test_2_minmag(deviceQueue, T{-1, -1, -1, -1}, T{2, 2, 2, 2},
                     T{-3, -3, -3, -3}, 2);
  math_test_2_remainder(deviceQueue, T{-1, -1, -1, -1}, T{5, 5, 5, 5},
                        T{2, 2, 2, 2}, 1);
  math_test_2_fdim(deviceQueue, T{-1, -1, -1, -1}, T{3, 3, 3, 3}, T{3, 3, 3, 3},
                   0);
  math_test_2_fmod(deviceQueue, T{-1, -1, -1, -1}, T{5, 5, 5, 5}, T{3, 3, 3, 3},
                   2);
  math_test_2_nextafter(deviceQueue, T{-1, -1, -1, -1}, T{-0, -0, -0, -0},
                        T{+0, +0, +0, +0}, 0);
  math_test_3_fma(deviceQueue, T{-1, -1, -1, -1}, T{2, 2, 2, 2}, T{2, 2, 2, 2},
                  T{1, 1, 1, 1}, 5);
  math_test_3_mad(deviceQueue, T{-1, -1, -1, -1}, T{2, 2, 2, 2}, T{2, 2, 2, 2},
                  T{1, 1, 1, 1}, 5);
  math_test_3_mix(deviceQueue, T{-1, -1, -1, -1}, T{3, 3, 3, 3}, T{5, 5, 5, 5},
                  T{0.5, 0.5, 0.5, 0.5}, 4);
}

template <typename T> void math_tests_3(queue &deviceQueue) {
  math_test_tan(deviceQueue, T{-1, -1, -1}, T{0, 0, 0}, 0);
  math_test_tanh(deviceQueue, T{-1, -1, -1}, T{0, 0, 0}, 0);
  math_test_cos(deviceQueue, T{-1, -1, -1}, T{0, 0, 0}, 1);
  math_test_sin(deviceQueue, T{-1, -1, -1}, T{0, 0, 0}, 0);
  math_test_cosh(deviceQueue, T{-1, -1, -1}, T{0, 0, 0}, 1);
  math_test_sinh(deviceQueue, T{-1, -1, -1}, T{0, 0, 0}, 0);
  math_test_acos(deviceQueue, T{-1, -1, -1}, T{1, 1, 1}, 0);
  math_test_acosh(deviceQueue, T{-1, -1, -1}, T{1, 1, 1}, 0);
  math_test_asin(deviceQueue, T{-1, -1, -1}, T{0, 0, 0}, 0);
  math_test_asinh(deviceQueue, T{-1, -1, -1}, T{0, 0, 0}, 0);
  math_test_cbrt(deviceQueue, T{-1, -1, -1}, T{1, 1, 1}, 1);
  math_test_atan(deviceQueue, T{-1, -1, -1}, T{0, 0, 0}, 0);
  math_test_atanh(deviceQueue, T{-1, -1, -1}, T{0, 0, 0}, 0);
  math_test_exp(deviceQueue, T{-1, -1, -1}, T{0, 0, 0}, 1);
  math_test_exp2(deviceQueue, T{-1, -1, -1}, T{2, 2, 2}, 4);
  math_test_exp10(deviceQueue, T{-1, -1, -1}, T{2, 2, 2}, 100);
  math_test_expm1(deviceQueue, T{-1, -1, -1}, T{0, 0, 0}, 0);
  math_test_ceil(deviceQueue, T{-1, -1, -1}, T{0.6, 0.6, 0.6}, 1);
  math_test_tgamma(deviceQueue, T{-1, -1, -1}, T{1, 1, 1}, 1);
  math_test_lgamma(deviceQueue, T{-1, -1, -1}, T{1, 1, 1}, 0);
  math_test_erf(deviceQueue, T{-1, -1, -1}, T{0, 0, 0}, 0);
  math_test_erfc(deviceQueue, T{-1, -1, -1}, T{0, 0, 0}, 1);
  math_test_log(deviceQueue, T{-1, -1, -1}, T{1, 1, 1}, 0);
  math_test_log2(deviceQueue, T{-1, -1, -1}, T{4, 4, 4}, 2);
  math_test_log10(deviceQueue, T{-1, -1, -1}, T{100, 100, 100}, 2);
  math_test_log1p(deviceQueue, T{-1, -1, -1}, T{0, 0, 0}, 0);
  math_test_logb(deviceQueue, T{-1, -1, -1}, T{1.1, 1.1, 1.1}, 0);
  math_test_sqrt(deviceQueue, T{-1, -1, -1}, T{4, 4, 4}, 2);
  math_test_rsqrt(deviceQueue, T{-1, -1, -1}, T{0.25, 0.25, 0.25}, 2);
  math_test_rint(deviceQueue, T{-1, -1, -1}, T{2.9, 2.9, 2.9}, 3);
  math_test_round(deviceQueue, T{-1, -1, -1}, T{0.5, 0.5, 0.5}, 1);
  math_test_trunc(deviceQueue, T{-1, -1, -1}, T{1.9, 1.9, 1.9}, 1);
}
