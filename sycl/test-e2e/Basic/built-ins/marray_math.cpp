// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}

// RUN: %{build} %{mathflags} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} %{mathflags} -fpreview-breaking-changes -o %t_preview.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t_preview.out%}

#include <cmath>
#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>

// Reference
// https://github.com/KhronosGroup/SYCL-CTS/blob/SYCL-2020/util/accuracy.h
template <typename T> T get_ulp_sycl(T x) {
  const T inf = std::numeric_limits<T>::infinity();
  const T negative = sycl::fabs(sycl::nextafter(x, -inf) - x);
  const T positive = sycl::fabs(sycl::nextafter(x, inf) - x);
  return sycl::fmin(negative, positive);
}

// Case 1: According to the SYCL 2020 specification, the sycl::half_precision
// math functions all take float or either an marray, vec, or swizzled vec of
// float. Additionally, the ULP expectation of all of these should be 8192
// according to the spec.
//
// Case 2: The allowed error in ULP is less than 8192 for half precision
// floating-point (sycl::half)
template <typename T> T get_8192ulp_sycl(T x) {
  const auto ulp = get_ulp_sycl<float>(x);
  const float multiplier = 8192.0f;
  return static_cast<T>(ulp * multiplier);
}

template <typename T> bool compare_floats_8192ulp(T actual, T expected) {
  const T difference = static_cast<T>(std::fabs(actual - expected));
  const T difference_expected = get_8192ulp_sycl(expected);
  return difference <= difference_expected;
}

#define TEST(FUNC, MARRAY_ELEM_TYPE, DIM, EXPECTED, DELTA, ...)                \
  {                                                                            \
    {                                                                          \
      MARRAY_ELEM_TYPE result[DIM];                                            \
      {                                                                        \
        sycl::buffer<MARRAY_ELEM_TYPE> b(result, sycl::range{DIM});            \
        deviceQueue.submit([&](sycl::handler &cgh) {                           \
          sycl::accessor res_access{b, cgh};                                   \
          cgh.single_task([=]() {                                              \
            sycl::marray<MARRAY_ELEM_TYPE, DIM> res = FUNC(__VA_ARGS__);       \
            for (int i = 0; i < DIM; i++)                                      \
              res_access[i] = res[i];                                          \
          });                                                                  \
        });                                                                    \
      }                                                                        \
      for (int i = 0; i < DIM; i++)                                            \
        assert(abs(result[i] - EXPECTED[i]) <= DELTA);                         \
    }                                                                          \
  }

#define TEST2(FUNC, MARRAY_ELEM_TYPE, PTR_TYPE, DIM, EXPECTED_1, EXPECTED_2,   \
              DELTA, ...)                                                      \
  {                                                                            \
    {                                                                          \
      MARRAY_ELEM_TYPE result[DIM];                                            \
      sycl::marray<PTR_TYPE, DIM> result_ptr;                                  \
      {                                                                        \
        sycl::buffer<MARRAY_ELEM_TYPE> b(result, sycl::range{DIM});            \
        sycl::buffer<sycl::marray<PTR_TYPE, DIM>, 1> b_ptr(&result_ptr,        \
                                                           sycl::range<1>(1)); \
        deviceQueue.submit([&](sycl::handler &cgh) {                           \
          sycl::accessor res_access{b, cgh};                                   \
          sycl::accessor res_ptr_access{b_ptr, cgh};                           \
          cgh.single_task([=]() {                                              \
            sycl::multi_ptr<sycl::marray<PTR_TYPE, DIM>,                       \
                            sycl::access::address_space::global_space,         \
                            sycl::access::decorated::no>                       \
                ptr(res_ptr_access);                                           \
            sycl::marray<MARRAY_ELEM_TYPE, DIM> res = FUNC(__VA_ARGS__, ptr);  \
            for (int i = 0; i < DIM; i++)                                      \
              res_access[i] = res[i];                                          \
          });                                                                  \
        });                                                                    \
      }                                                                        \
      for (int i = 0; i < DIM; i++) {                                          \
        assert(std::abs(result[i] - EXPECTED_1[i]) <= DELTA);                  \
        assert(std::abs(result_ptr[i] - EXPECTED_2[i]) <= DELTA);              \
      }                                                                        \
    }                                                                          \
  }

#define TEST3(FUNC, MARRAY_ELEM_TYPE, DIM, ...)                                \
  {                                                                            \
    {                                                                          \
      MARRAY_ELEM_TYPE result[DIM];                                            \
      {                                                                        \
        sycl::buffer<MARRAY_ELEM_TYPE> b(result, sycl::range{DIM});            \
        deviceQueue.submit([&](sycl::handler &cgh) {                           \
          sycl::accessor res_access{b, cgh};                                   \
          cgh.single_task([=]() {                                              \
            sycl::marray<MARRAY_ELEM_TYPE, DIM> res = FUNC(__VA_ARGS__);       \
            for (int i = 0; i < DIM; i++)                                      \
              res_access[i] = res[i];                                          \
          });                                                                  \
        });                                                                    \
      }                                                                        \
    }                                                                          \
  }

#define TEST4(FUNC, MARRAY_ELEM_TYPE, DIM, EXPECTED, ...)                      \
  {                                                                            \
    {                                                                          \
      MARRAY_ELEM_TYPE result[DIM];                                            \
      {                                                                        \
        sycl::buffer<MARRAY_ELEM_TYPE> b(result, sycl::range{DIM});            \
        deviceQueue.submit([&](sycl::handler &cgh) {                           \
          sycl::accessor res_access{b, cgh};                                   \
          cgh.single_task([=]() {                                              \
            sycl::marray<MARRAY_ELEM_TYPE, DIM> res = FUNC(__VA_ARGS__);       \
            for (int i = 0; i < DIM; i++)                                      \
              res_access[i] = res[i];                                          \
          });                                                                  \
        });                                                                    \
      }                                                                        \
      for (int i = 0; i < DIM; i++)                                            \
        assert(compare_floats_8192ulp(result[i], EXPECTED[i]));                \
    }                                                                          \
  }

#define EXPECTED(TYPE, ...) ((TYPE[]){__VA_ARGS__})

int main() {
  sycl::queue deviceQueue;

  sycl::marray<float, 2> ma1{1.0f, 2.0f};
  sycl::marray<float, 2> ma2{3.0f, 2.0f};
  sycl::marray<float, 3> ma3{180, 180, 180};
  sycl::marray<int, 3> ma4{1, 1, 1};
  sycl::marray<float, 3> ma5{180, -180, -180};
  sycl::marray<float, 3> ma6{1.4f, 4.2f, 5.3f};
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  sycl::marray<unsigned int, 3> ma7{1, 2, 3};
  sycl::marray<unsigned long int, 3> ma8{1, 2, 3};
#else
  sycl::marray<uint32_t, 3> ma7{1, 2, 3};
  sycl::marray<uint64_t, 3> ma8{1, 2, 3};
#endif

  TEST(sycl::fabs, float, 3, EXPECTED(float, 180, 180, 180), 0, ma5);
  TEST(sycl::ilogb, int, 3, EXPECTED(int, 7, 7, 7), 0, ma3);
  TEST(sycl::fmax, float, 2, EXPECTED(float, 3.0f, 2.0f), 0, ma1, ma2);
  TEST(sycl::fmax, float, 2, EXPECTED(float, 5.0f, 5.0f), 0, ma1, 5.0f);
  TEST(sycl::fmin, float, 2, EXPECTED(float, 1.0f, 2.0f), 0, ma1, ma2);
  TEST(sycl::fmin, float, 2, EXPECTED(float, 1.0f, 2.0f), 0, ma1, 5.0f);
  TEST(sycl::ldexp, float, 3, EXPECTED(float, 360, 360, 360), 0, ma3, ma4);
  TEST(sycl::ldexp, float, 3, EXPECTED(float, 5760, 5760, 5760), 0, ma3, 5);
  TEST(sycl::pown, float, 3, EXPECTED(float, 180, 180, 180), 0.1, ma3, ma4);
  TEST(sycl::rootn, float, 3, EXPECTED(float, 180, 180, 180), 0.1, ma3, ma4);
  TEST2(sycl::fract, float, float, 3, EXPECTED(float, 0.4f, 0.2f, 0.3f),
        EXPECTED(float, 1, 4, 5), 0.0001, ma6);
  TEST2(sycl::modf, float, float, 3, EXPECTED(float, 0.4f, 0.2f, 0.3f),
        EXPECTED(float, 1, 4, 5), 0.0001, ma6);
  TEST2(sycl::sincos, float, float, 3,
        EXPECTED(float, 0.98545f, -0.871576f, -0.832267f),
        EXPECTED(float, 0.169967, -0.490261, 0.554375), 0.0001, ma6);
  TEST2(sycl::frexp, float, int, 3, EXPECTED(float, 0.7f, 0.525f, 0.6625f),
        EXPECTED(int, 1, 3, 3), 0.0001, ma6);
  TEST2(sycl::lgamma_r, float, int, 3,
        EXPECTED(float, -0.119613f, 2.04856f, 3.63964f), EXPECTED(int, 1, 1, 1),
        0.0001, ma6);
  TEST2(sycl::remquo, float, int, 3, EXPECTED(float, 1.4f, 4.2f, 5.3f),
        EXPECTED(int, 0, 0, 0), 0.0001, ma6, ma3);
  TEST3(sycl::nan, float, 3, ma7);
  if (deviceQueue.get_device().has(sycl::aspect::fp64))
    TEST3(sycl::nan, double, 3, ma8);

  TEST4(sycl::half_precision::sin, float, 3,
        EXPECTED(float, 0.98545f, -0.871576f, -0.832267f), ma6);
  TEST4(sycl::half_precision::cos, float, 3,
        EXPECTED(float, 0.169967f, -0.490261f, 0.554375f), ma6);
  TEST4(sycl::half_precision::tan, float, 3,
        EXPECTED(float, 5.797f, 1.777f, -1.501f), ma6);
  TEST4(sycl::half_precision::divide, float, 2, EXPECTED(float, 3.0f, 1.0f),
        ma2, ma1);
  TEST4(sycl::half_precision::log, float, 2, EXPECTED(float, 0.0f, 0.693f),
        ma1);
  TEST4(sycl::half_precision::log2, float, 2, EXPECTED(float, 0.0f, 1.f), ma1);
  TEST4(sycl::half_precision::log10, float, 2, EXPECTED(float, 0.0f, 0.301f),
        ma1);
  TEST4(sycl::half_precision::powr, float, 2, EXPECTED(float, 3.0f, 4.0f), ma2,
        ma1);
  TEST4(sycl::half_precision::recip, float, 2, EXPECTED(float, 0.333333f, 0.5f),
        ma2);
  TEST4(sycl::half_precision::sqrt, float, 2, EXPECTED(float, 1.0f, 1.414f),
        ma1);
  TEST4(sycl::half_precision::rsqrt, float, 2, EXPECTED(float, 1.0f, 0.707f),
        ma1);
  TEST4(sycl::half_precision::exp, float, 2, EXPECTED(float, 2.718f, 7.389f),
        ma1);
  TEST4(sycl::half_precision::exp2, float, 2, EXPECTED(float, 2, 4), ma1);
  TEST4(sycl::half_precision::exp10, float, 2, EXPECTED(float, 10, 100), ma1);

  return 0;
}
