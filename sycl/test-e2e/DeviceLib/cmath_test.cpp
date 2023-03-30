// UNSUPPORTED: hip
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fno-builtin %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// RUN: %clangxx -fsycl -fno-builtin -fsycl-device-lib-jit-link %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include "math_utils.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sycl/sycl.hpp>

namespace s = sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

#define TEST_NUM 59

float ref[TEST_NUM] = {1, 0, 0,   0,   0,   0,   0, 1, 1, 0.5, 0, 0, 1, 0, 2,
                       0, 0, 0,   0,   0,   1,   0, 1, 2, 0,   1, 2, 5, 0, 0,
                       0, 0, 0.5, 0.5, NAN, NAN, 2, 0, 0, 0,   0, 0, 0, 0, 0,
                       0, 0, 0,   0,   0,   0,   0, 0, 0, 0,   0, 0, 0, 0};

float refIptr = 1;

template <class T> void device_cmath_test_1(s::queue &deviceQueue) {
  s::range<1> numOfItems{TEST_NUM};
  T result[TEST_NUM] = {-1};

  // Variable iptr stores the integral part of float point in modf function
  T iptr = -1;

  // Variable quo stores the sign and some bits of x/y in remquo function
  int quo = -1;
  {
    s::buffer<T, 1> buffer1(result, numOfItems);
    s::buffer<T, 1> buffer2(&iptr, s::range<1>{1});
    s::buffer<int, 1> buffer3(&quo, s::range<1>{1});
    deviceQueue.submit([&](s::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      auto iptr_access = buffer2.template get_access<sycl_write>(cgh);
      auto quo_access = buffer3.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceMathTest1>([=]() {
        int i = 0;
        T nan = NAN;
        T minus_nan = -NAN;
        T infinity = INFINITY;
        T minus_infinity = -INFINITY;
        float subnormal;
        *((uint32_t *)&subnormal) = 0x7FFFFF;

        res_access[i++] = std::cos(0.0f);
        res_access[i++] = std::sin(0.0f);
        res_access[i++] = std::log(1.0f);
        res_access[i++] = std::acos(1.0f);
        res_access[i++] = std::asin(0.0f);
        res_access[i++] = std::atan(0.0f);
        res_access[i++] = std::atan2(0.0f, 1.0f);
        res_access[i++] = std::cosh(0.0f);
        res_access[i++] = std::exp(0.0f);
        res_access[i++] = std::fmod(1.5f, 1.0f);
        res_access[i++] = std::log10(1.0f);
        res_access[i++] = std::modf(1.0f, &iptr_access[0]);
        res_access[i++] = std::pow(1.0f, 1.0f);
        res_access[i++] = std::sinh(0.0f);
        res_access[i++] = std::sqrt(4.0f);
        res_access[i++] = std::tan(0.0f);
        res_access[i++] = std::tanh(0.0f);
        res_access[i++] = std::acosh(1.0f);
        res_access[i++] = std::asinh(0.0f);
        res_access[i++] = std::atanh(0.0f);
        res_access[i++] = std::cbrt(1.0f);
        res_access[i++] = std::erf(0.0f);
        res_access[i++] = std::erfc(0.0f);
        res_access[i++] = std::exp2(1.0f);
        res_access[i++] = std::expm1(0.0f);
        res_access[i++] = std::fdim(1.0f, 0.0f);
        res_access[i++] = std::fma(1.0f, 1.0f, 1.0f);
        res_access[i++] = std::hypot(3.0f, 4.0f);
        res_access[i++] = std::ilogb(1.0f);
        res_access[i++] = std::log1p(0.0f);
        res_access[i++] = std::log2(1.0f);
        res_access[i++] = std::logb(1.0f);
        res_access[i++] = std::remainder(0.5f, 1.0f);
        res_access[i++] = std::remquo(0.5f, 1.0f, &quo_access[0]);
        res_access[i++] = std::tgamma(nan);
        res_access[i++] = std::lgamma(nan);
        res_access[i++] = std::scalbn(1.0f, 1);

        res_access[i++] = !(std::signbit(infinity) == 0);
        res_access[i++] = !(std::signbit(minus_infinity) != 0);
        res_access[i++] = !(std::isunordered(minus_nan, nan) != 0);
        res_access[i++] = !(std::isunordered(minus_infinity, infinity) == 0);
        res_access[i++] = !(std::isgreater(minus_infinity, infinity) == 0);
        res_access[i++] = !(std::isgreater(0.0f, minus_nan) == 0);
        res_access[i++] = !(std::isfinite(0.0f) != 0);
        res_access[i++] = !(std::isfinite(nan) == 0);
        res_access[i++] = !(std::isfinite(infinity) == 0);
        res_access[i++] = !(std::isfinite(minus_infinity) == 0);

        res_access[i++] = !(std::isinf(0.0f) == 0);
        res_access[i++] = !(std::isinf(nan) == 0);
        res_access[i++] = !(std::isinf(infinity) != 0);
        res_access[i++] = !(std::isinf(minus_infinity) != 0);
        res_access[i++] = !(std::isnan(0.0f) == 0);
        res_access[i++] = !(std::isnan(nan) != 0);
        res_access[i++] = !(std::isnan(infinity) == 0);
        res_access[i++] = !(std::isnan(minus_infinity) == 0);
        res_access[i++] = !(std::isnormal(nan) == 0);
        res_access[i++] = !(std::isnormal(minus_infinity) == 0);
        res_access[i++] = !(std::isnormal(subnormal) == 0);
        res_access[i++] = !(std::isnormal(1.0f) != 0);
      });
    });
  }

  // Compare result with reference
  for (int i = 0; i < TEST_NUM; ++i) {
    assert(approx_equal_fp(result[i], ref[i]));
  }

  // Test modf integral part
  assert(approx_equal_fp(iptr, refIptr));

  // Test remquo sign
  assert(quo == 0);
}

// MSVC implements std::ldexp<float> and std::frexp<float> by invoking the
// 'double' version of corresponding C math functions(ldexp and frexp). Those
// 2 functions can only work on Windows with fp64 extension support from
// underlying device.
#ifndef _WIN32
template <class T> void device_cmath_test_2(s::queue &deviceQueue) {
  s::range<1> numOfItems{2};
  T result[2] = {-1};
  T ref[2] = {0, 2};
  // Variable exponent is an integer value to store the exponent in frexp
  // function
  int exponent = -1;

  {
    s::buffer<T, 1> buffer1(result, numOfItems);
    s::buffer<int, 1> buffer2(&exponent, s::range<1>{1});
    deviceQueue.submit([&](s::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      auto exp_access = buffer2.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceMathTest2>([=]() {
        int i = 0;
        res_access[i++] = std::frexp(0.0f, &exp_access[0]);
        res_access[i++] = std::ldexp(1.0f, 1);
      });
    });
  }

  // Compare result with reference
  for (int i = 0; i < 2; ++i) {
    assert(approx_equal_fp(result[i], ref[i]));
  }

  // Test frexp exponent
  assert(exponent == 0);
}
#endif

// Disable std::lldiv and std::ldiv test since it will lead
// to OCL CPU failure, OCL CPU runtime has already fixed but
// need to wait for the fix to propagate into pre-ci environment.
void device_integer_math_test(s::queue &deviceQueue) {
  div_t result_i[1];
  // ldiv_t result_l[1];
  // lldiv_t result_ll[1];

  int result_i2[1];
  long int result_l2[1];
  long long int result_ll2[1];

  {
    s::buffer<div_t, 1> buffer1(result_i, s::range<1>{1});
    // s::buffer<ldiv_t, 1> buffer2(result_l, s::range<1>{1});
    // s::buffer<lldiv_t, 1> buffer3(result_ll, s::range<1>{1});
    s::buffer<int, 1> buffer4(result_i2, s::range<1>{1});
    s::buffer<long int, 1> buffer5(result_l2, s::range<1>{1});
    s::buffer<long long int, 1> buffer6(result_ll2, s::range<1>{1});
    deviceQueue.submit([&](s::handler &cgh) {
      auto res_i_access = buffer1.get_access<sycl_write>(cgh);
      // auto res_l_access = buffer2.get_access<sycl_write>(cgh);
      // auto res_ll_access = buffer3.get_access<sycl_write>(cgh);
      auto res_i2_access = buffer4.get_access<sycl_write>(cgh);
      auto res_l2_access = buffer5.get_access<sycl_write>(cgh);
      auto res_ll2_access = buffer6.get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceIntMathTest>([=]() {
        res_i_access[0] = std::div(99, 4);
        // res_l_access[0] = std::ldiv(10000, 23);
        // res_ll_access[0] = std::lldiv(200000000, 47);
        res_i2_access[0] = std::abs(-111);
        res_l2_access[0] = std::labs(10000);
        res_ll2_access[0] = std::llabs(-2000000);
      });
    });
  }

  assert(result_i[0].quot == 24 && result_i[0].rem == 3);
  // assert(result_l[0].quot == 434 && result_l[0].rem == 18);
  // assert(result_ll[0].quot == 4255319 && result_ll[0].rem == 7);
  assert(result_i2[0] == 111);
  assert(result_l2[0] == 10000);
  assert(result_ll2[0] == 2000000);
}

int main() {
  s::queue deviceQueue;
  device_cmath_test_1<float>(deviceQueue);
#ifndef _WIN32
  device_cmath_test_2<float>(deviceQueue);
#endif
  device_integer_math_test(deviceQueue);
  std::cout << "Pass" << std::endl;
  return 0;
}
