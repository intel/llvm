// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// RUN: %clangxx -fsycl -fsycl-device-lib-jit-link %s -o %t.out
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

#define TEST_NUM 61

double ref[TEST_NUM] = {
    1, 0, 0, 0, 0, 0, 0, 1, 1, 0.5, 0, 2, 0, 0,   1,   0,   2,   0, 0, 0, 0,
    0, 1, 0, 1, 2, 0, 1, 2, 5, 0,   0, 0, 0, 0.5, 0.5, NAN, NAN, 2, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0,   0,   0,   0,   0, 0};

double refIptr = 1;

template <class T> void device_cmath_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{TEST_NUM};
  T result[TEST_NUM] = {-1};

  // Variable exponent is an integer value to store the exponent in frexp
  // function
  int exponent = -1;

  // Variable iptr stores the integral part of float point in modf function
  T iptr = -1;

  // Variable quo stores the sign and some bits of x/y in remquo function
  int quo = -1;
  {
    s::buffer<T, 1> buffer1(result, numOfItems);
    s::buffer<int, 1> buffer2(&exponent, s::range<1>{1});
    s::buffer<T, 1> buffer3(&iptr, s::range<1>{1});
    s::buffer<int, 1> buffer4(&quo, s::range<1>{1});
    deviceQueue.submit([&](sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      auto exp_access = buffer2.template get_access<sycl_write>(cgh);
      auto iptr_access = buffer3.template get_access<sycl_write>(cgh);
      auto quo_access = buffer4.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceMathTest>([=]() {
        int i = 0;
        T nan = NAN;
        T minus_nan = -NAN;
        T infinity = INFINITY;
        T minus_infinity = -INFINITY;
        double subnormal;
        *((uint64_t *)&subnormal) = 0xFFFFFFFFFFFFFULL;
        res_access[i++] = std::cos(0.0);
        res_access[i++] = std::sin(0.0);
        res_access[i++] = std::log(1.0);
        res_access[i++] = std::acos(1.0);
        res_access[i++] = std::asin(0.0);
        res_access[i++] = std::atan(0.0);
        res_access[i++] = std::atan2(0.0, 1.0);
        res_access[i++] = std::cosh(0.0);
        res_access[i++] = std::exp(0.0);
        res_access[i++] = std::fmod(1.5, 1.0);
        res_access[i++] = std::frexp(0.0, &exp_access[0]);
        res_access[i++] = std::ldexp(1.0, 1);
        res_access[i++] = std::log10(1.0);
        res_access[i++] = std::modf(1.0, &iptr_access[0]);
        res_access[i++] = std::pow(1.0, 1.0);
        res_access[i++] = std::sinh(0.0);
        res_access[i++] = std::sqrt(4.0);
        res_access[i++] = std::tan(0.0);
        res_access[i++] = std::tanh(0.0);
        res_access[i++] = std::acosh(1.0);
        res_access[i++] = std::asinh(0.0);
        res_access[i++] = std::atanh(0.0);
        res_access[i++] = std::cbrt(1.0);
        res_access[i++] = std::erf(0.0);
        res_access[i++] = std::erfc(0.0);
        res_access[i++] = std::exp2(1.0);
        res_access[i++] = std::expm1(0.0);
        res_access[i++] = std::fdim(1.0, 0.0);
        res_access[i++] = std::fma(1.0, 1.0, 1.0);
        res_access[i++] = std::hypot(3.0, 4.0);
        res_access[i++] = std::ilogb(1.0);
        res_access[i++] = std::log1p(0.0);
        res_access[i++] = std::log2(1.0);
        res_access[i++] = std::logb(1.0);
        res_access[i++] = std::remainder(0.5, 1.0);
        res_access[i++] = std::remquo(0.5, 1.0, &quo_access[0]);
        res_access[i++] = std::tgamma(nan);
        res_access[i++] = std::lgamma(nan);
        res_access[i++] = std::scalbn(1.0, 1);

        res_access[i++] = !(std::signbit(infinity) == 0);
        res_access[i++] = !(std::signbit(minus_infinity) != 0);
        res_access[i++] = !(std::isunordered(minus_nan, nan) != 0);
        res_access[i++] = !(std::isunordered(minus_infinity, infinity) == 0);
        res_access[i++] = !(std::isgreater(minus_infinity, infinity) == 0);
        res_access[i++] = !(std::isgreater(0.0f, minus_nan) == 0);
#ifdef _WIN32
        res_access[i++] = !(std::isfinite(0.0f) != 0);
        res_access[i++] = !(std::isfinite(nan) == 0);
        res_access[i++] = !(std::isfinite(infinity) == 0);
        res_access[i++] = !(std::isfinite(minus_infinity) == 0);

        res_access[i++] = !(std::isinf(0.0f) == 0);
        res_access[i++] = !(std::isinf(nan) == 0);
        res_access[i++] = !(std::isinf(infinity) != 0);
        res_access[i++] = !(std::isinf(minus_infinity) != 0);
#else  // !_WIN32
       // __builtin_isfinite is unsupported.
        res_access[i++] = 0;
        res_access[i++] = 0;
        res_access[i++] = 0;
        res_access[i++] = 0;

        // __builtin_isinf is unsupported.
        res_access[i++] = 0;
        res_access[i++] = 0;
        res_access[i++] = 0;
        res_access[i++] = 0;
#endif // !_WIN32
        res_access[i++] = !(std::isnan(0.0f) == 0);
        res_access[i++] = !(std::isnan(nan) != 0);
        res_access[i++] = !(std::isnan(infinity) == 0);
        res_access[i++] = !(std::isnan(minus_infinity) == 0);
#ifdef _WIN32
        res_access[i++] = !(std::isnormal(nan) == 0);
        res_access[i++] = !(std::isnormal(minus_infinity) == 0);
        res_access[i++] = !(std::isnormal(subnormal) == 0);
        res_access[i++] = !(std::isnormal(1.0f) != 0);
#else  // !_WIN32
       // __builtin_isnormal() is unsupported.
        res_access[i++] = 0;
        res_access[i++] = 0;
        res_access[i++] = 0;
        res_access[i++] = 0;
#endif // !_WIN32
      });
    });
  }

  // Compare result with reference
  for (int i = 0; i < TEST_NUM; ++i) {
    assert(approx_equal_fp(result[i], ref[i]));
  }

  // Test modf integral part
  assert(approx_equal_fp(iptr, refIptr));

  // Test frexp exponent
  assert(exponent == 0);

  // Test remquo sign
  assert(quo == 0);
}

int main() {
  s::queue deviceQueue;
  if (deviceQueue.get_device().has(sycl::aspect::fp64)) {
    device_cmath_test<double>(deviceQueue);
    std::cout << "Pass" << std::endl;
  }
  return 0;
}
