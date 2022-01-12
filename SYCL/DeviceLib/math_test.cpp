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

#define TEST_NUM 59

float ref_val[TEST_NUM] = {
    1, 0, 0, 0, 0, 0, 0, 1, 1, 0.5, 0, 0, 1,   0,   2,   0,   0, 0, 0, 0,
    1, 0, 1, 2, 0, 1, 2, 5, 0, 0,   0, 0, 0.5, 0.5, NAN, NAN, 2, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0,   0,   0,   0,   0, 0, 0};

float refIptr = 1;

void device_math_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{TEST_NUM};
  float result[TEST_NUM] = {-1};

  // Variable iptr stores the integral part of float point in modf function
  float iptr = -1;

  // Variable quo stores the sign and some bits of x/y in remquo function
  int quo = -1;
  {
    s::buffer<float, 1> buffer1(result, numOfItems);
    s::buffer<float, 1> buffer2(&iptr, s::range<1>{1});
    s::buffer<int, 1> buffer3(&quo, s::range<1>{1});
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      auto iptr_access = buffer2.template get_access<sycl_write>(cgh);
      auto quo_access = buffer3.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceMathTest>([=]() {
        int i = 0;
        float nan = NAN;
        float minus_nan = -NAN;
        float infinity = INFINITY;
        float minus_infinity = -INFINITY;
        float subnormal;
        *((uint32_t *)&subnormal) = 0x7FFFFF;

        res_access[i++] = cosf(0.0f);
        res_access[i++] = sinf(0.0f);
        res_access[i++] = logf(1.0f);
        res_access[i++] = acosf(1.0f);
        res_access[i++] = asinf(0.0f);
        res_access[i++] = atanf(0.0f);
        res_access[i++] = atan2f(0.0f, 1.0f);
        res_access[i++] = coshf(0.0f);
        res_access[i++] = expf(0.0f);
        res_access[i++] = fmodf(1.5f, 1.0f);
        res_access[i++] = log10f(1.0f);
        res_access[i++] = modff(1.0f, &iptr_access[0]);
        res_access[i++] = powf(1.0f, 1.0f);
        res_access[i++] = sinhf(0.0f);
        res_access[i++] = sqrtf(4.0f);
        res_access[i++] = tanf(0.0f);
        res_access[i++] = tanhf(0.0f);
        res_access[i++] = acoshf(1.0f);
        res_access[i++] = asinhf(0.0f);
        res_access[i++] = atanhf(0.0f);
        res_access[i++] = cbrtf(1.0f);
        res_access[i++] = erff(0.0f);
        res_access[i++] = erfcf(0.0f);
        res_access[i++] = exp2f(1.0f);
        res_access[i++] = expm1f(0.0f);
        res_access[i++] = fdimf(1.0f, 0.0f);
        res_access[i++] = fmaf(1.0f, 1.0f, 1.0f);
        res_access[i++] = hypotf(3.0f, 4.0f);
        res_access[i++] = ilogbf(1.0f);
        res_access[i++] = log1pf(0.0f);
        res_access[i++] = log2f(1.0f);
        res_access[i++] = logbf(1.0f);
        res_access[i++] = remainderf(0.5f, 1.0f);
        res_access[i++] = remquof(0.5f, 1.0f, &quo_access[0]);
        res_access[i++] = tgammaf(nan);
        res_access[i++] = lgammaf(nan);
        res_access[i++] = scalbnf(1.0f, 1);

        res_access[i++] = !(signbit(infinity) == 0);
        res_access[i++] = !(signbit(minus_infinity) != 0);
        res_access[i++] = !(isunordered(minus_nan, nan) != 0);
        res_access[i++] = !(isunordered(minus_infinity, infinity) == 0);
        res_access[i++] = !(isgreater(minus_infinity, infinity) == 0);
        res_access[i++] = !(isgreater(0.0f, minus_nan) == 0);
#ifdef _WIN32
        res_access[i++] = !(isfinite(0.0f) != 0);
        res_access[i++] = !(isfinite(nan) == 0);
        res_access[i++] = !(isfinite(infinity) == 0);
        res_access[i++] = !(isfinite(minus_infinity) == 0);

        res_access[i++] = !(isinf(0.0f) == 0);
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
        res_access[i++] = !(isnan(0.0f) == 0);
        res_access[i++] = !(isnan(nan) != 0);
        res_access[i++] = !(isnan(infinity) == 0);
        res_access[i++] = !(isnan(minus_infinity) == 0);
#ifdef _WIN32
        res_access[i++] = !(isnormal(nan) == 0);
        res_access[i++] = !(isnormal(minus_infinity) == 0);
        res_access[i++] = !(isnormal(subnormal) == 0);
        res_access[i++] = !(isnormal(1.0f) != 0);
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

  // Test remquo sign
  assert(quo == 0);
}

int main() {
  s::queue deviceQueue;
  device_math_test(deviceQueue);
  std::cout << "Pass" << std::endl;
  return 0;
}
