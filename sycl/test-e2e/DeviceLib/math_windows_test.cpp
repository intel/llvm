// REQUIRES: windows
// REQUIRES: build-and-run-mode

// TODO: Add hypotf case back when the missing symbol is fixed.

// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}

// RUN: %{build} %{mathflags} -o %t.out
// RUN: %{run} %t.out

// RUN: %clangxx -fsycl -fsycl-device-lib-jit-link %{mathflags} %s -o %t.out
// RUN: %if !gpu %{ %{run} %t.out %}

#include "math_utils.hpp"
#include <iostream>
#include <math.h>
#include <sycl/detail/core.hpp>

namespace s = sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

#define TEST_NUM 35

float ref_val[TEST_NUM] = {1, 0, 0, 0, 0, 0, 0, 1,   1,   0.5, 0,  0,
                           1, 0, 2, 0, 0, 0, 0, 0,   1,   0,   1,  2,
                           0, 1, 2, 0, 0, 0, 0, 0.5, 0.5, NAN, NAN};

float refIptr = 1;

void device_math_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{TEST_NUM};
  float result[TEST_NUM] = {-1};

  // Variable iptr stores the integral part of float point in modf function
  float iptr = -1;

  // Variable quo stores the sign and some bits of x/y in remquo function
  int quo = -1;

  // Varaible enm stores the enum value retured by MSVC function
  short enm = 10;

  {
    s::buffer<float, 1> buffer1(result, numOfItems);
    s::buffer<float, 1> buffer2(&iptr, s::range<1>{1});
    s::buffer<int, 1> buffer3(&quo, s::range<1>{1});
    s::buffer<short, 1> buffer4(&enm, s::range<1>{1});
    deviceQueue.submit([&](sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      auto iptr_access = buffer2.template get_access<sycl_write>(cgh);
      auto quo_access = buffer3.template get_access<sycl_write>(cgh);
      auto enm_access = buffer4.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceMathTest>([=]() {
        int i = 0;
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
        res_access[i++] = ilogbf(1.0f);
        res_access[i++] = log1pf(0.0f);
        res_access[i++] = log2f(1.0f);
        res_access[i++] = logbf(1.0f);
        res_access[i++] = remainderf(0.5f, 1.0f);
        res_access[i++] = remquof(0.5f, 1.0f, &quo_access[0]);
        float a = NAN;
        res_access[i++] = tgammaf(a);
        res_access[i++] = lgammaf(a);
        enm_access[0] = _fdtest(&a);
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

  // Test enum value returned by _FDtest
  assert(enm == _NANCODE);
}

int main() {
  s::queue deviceQueue;
  device_math_test(deviceQueue);
  std::cout << "Pass" << std::endl;
  return 0;
}
