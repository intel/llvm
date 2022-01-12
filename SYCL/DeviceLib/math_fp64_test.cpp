// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include "math_utils.hpp"
#include <CL/sycl.hpp>
#include <cstdint>
#include <iostream>
#include <math.h>

namespace s = cl::sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

#define TEST_NUM 61

double ref_val[TEST_NUM] = {
    1, 0, 0, 0, 0, 0, 0, 1, 1, 0.5, 0, 2, 0, 0,   1,   0,   2,   0, 0, 0, 0,
    0, 1, 0, 1, 2, 0, 1, 2, 5, 0,   0, 0, 0, 0.5, 0.5, NAN, NAN, 2, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0,   0,   0,   0,   0, 0};

double refIptr = 1;

void device_math_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{TEST_NUM};
  double result[TEST_NUM] = {-1};

  // Variable exponent is an integer value to store the exponent in frexp
  // function
  int exponent = -1;

  // Variable iptr stores the integral part of float point in modf function
  double iptr = -1;

  // Variable quo stores the sign and some bits of x/y in remquo function
  int quo = -1;
  {
    s::buffer<double, 1> buffer1(result, numOfItems);
    s::buffer<int, 1> buffer2(&exponent, s::range<1>{1});
    s::buffer<double, 1> buffer3(&iptr, s::range<1>{1});
    s::buffer<int, 1> buffer4(&quo, s::range<1>{1});
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      auto exp_access = buffer2.template get_access<sycl_write>(cgh);
      auto iptr_access = buffer3.template get_access<sycl_write>(cgh);
      auto quo_access = buffer4.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceMathTest>([=]() {
        int i = 0;
        double nan = NAN;
        double minus_nan = -NAN;
        double infinity = INFINITY;
        double minus_infinity = -INFINITY;
        double subnormal;
        *((uint64_t *)&subnormal) = 0xFFFFFFFFFFFFFULL;

        res_access[i++] = cos(0.0);
        res_access[i++] = sin(0.0);
        res_access[i++] = log(1.0);
        res_access[i++] = acos(1.0);
        res_access[i++] = asin(0.0);
        res_access[i++] = atan(0.0);
        res_access[i++] = atan2(0.0, 1.0);
        res_access[i++] = cosh(0.0);
        res_access[i++] = exp(0.0);
        res_access[i++] = fmod(1.5, 1.0);
        res_access[i++] = frexp(0.0, &exp_access[0]);
        res_access[i++] = ldexp(1.0, 1);
        res_access[i++] = log10(1.0);
        res_access[i++] = modf(1.0, &iptr_access[0]);
        res_access[i++] = pow(1.0, 1.0);
        res_access[i++] = sinh(0.0);
        res_access[i++] = sqrt(4.0);
        res_access[i++] = tan(0.0);
        res_access[i++] = tanh(0.0);
        res_access[i++] = acosh(1.0);
        res_access[i++] = asinh(0.0);
        res_access[i++] = atanh(0.0);
        res_access[i++] = cbrt(1.0);
        res_access[i++] = erf(0.0);
        res_access[i++] = erfc(0.0);
        res_access[i++] = exp2(1.0);
        res_access[i++] = expm1(0.0);
        res_access[i++] = fdim(1.0, 0.0);
        res_access[i++] = fma(1.0, 1.0, 1.0);
        res_access[i++] = hypot(3.0, 4.0);
        res_access[i++] = ilogb(1.0);
        res_access[i++] = log1p(0.0);
        res_access[i++] = log2(1.0);
        res_access[i++] = logb(1.0);
        res_access[i++] = remainder(0.5, 1.0);
        res_access[i++] = remquo(0.5, 1.0, &quo_access[0]);
        res_access[i++] = tgamma(nan);
        res_access[i++] = lgamma(nan);
        res_access[i++] = scalbn(1.0, 1);

        res_access[i++] = !(signbit(infinity) == 0);
        res_access[i++] = !(signbit(minus_infinity) != 0);
        res_access[i++] = !(isunordered(minus_nan, nan) != 0);
        res_access[i++] = !(isunordered(minus_infinity, infinity) == 0);
        res_access[i++] = !(isgreater(minus_infinity, infinity) == 0);
        res_access[i++] = !(isgreater(0.0, minus_nan) == 0);
#ifdef _WIN32
        res_access[i++] = !(isfinite(0.0) != 0);
        res_access[i++] = !(isfinite(nan) == 0);
        res_access[i++] = !(isfinite(infinity) == 0);
        res_access[i++] = !(isfinite(minus_infinity) == 0);

        res_access[i++] = !(isinf(0.0) == 0);
        res_access[i++] = !(isinf(nan) == 0);
        res_access[i++] = !(isinf(infinity) != 0);
        res_access[i++] = !(isinf(minus_infinity) != 0);
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
        res_access[i++] = !(isnan(0.0) == 0);
        res_access[i++] = !(isnan(nan) != 0);
        res_access[i++] = !(isnan(infinity) == 0);
        res_access[i++] = !(isnan(minus_infinity) == 0);
#ifdef _WIN32
        res_access[i++] = !(isnormal(nan) == 0);
        res_access[i++] = !(isnormal(minus_infinity) == 0);
        res_access[i++] = !(isnormal(subnormal) == 0);
        res_access[i++] = !(isnormal(1.0) != 0);
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
    assert(approx_equal_fp(result[i], ref_val[i]));
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
    device_math_test(deviceQueue);
    std::cout << "Pass" << std::endl;
  }
  return 0;
}
