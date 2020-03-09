// UNSUPPORTED: windows
// RUN: %clangxx -fsycl -c %s -o %t.o
// RUN: %clangxx -fsycl %t.o %sycl_libs_dir/libsycl-cmath-fp64.o -o %t.out
#include <CL/sycl.hpp>
#include <iostream>
#include <math.h>
#include "math_utils.hpp"

namespace s = cl::sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

double a = 0;
double b = 1;
double c = 0.5;
double d = 2;
double e = 5;

void device_math_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{38};
  double result[38] = {-1};
  double ref[38] = {
      b,
      a,
      a,
      a,
      a,
      a,
      a,
      b,
      b,
      c,
      a,
      d,
      a,
      a,
      b,
      a,
      d,
      a,
      a,
      a,
      a,
      a,
      b,
      a,
      b,
      d,
      a,
      b,
      d,
      e,
      a,
      a,
      a,
      a,
      c,
      c,
      a,
      a,
  };
  int expv = -1;
  double iptr = -1;
  int quo = -1;
  {
    s::buffer<double, 1> buffer1(result, numOfItems);
    s::buffer<int, 1> buffer2(&expv, s::range<1>{1});
    s::buffer<double, 1> buffer3(&iptr, s::range<1>{1});
    s::buffer<int, 1> buffer4(&quo, s::range<1>{1});
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      auto exp_access = buffer2.template get_access<sycl_write>(cgh);
      auto iptr_access = buffer3.template get_access<sycl_write>(cgh);
      auto quo_access = buffer4.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceMathTest>([=]() {
        int i = 0;
        {
          double a = 0;
          res_access[i++] = cos(a);
        }
        {
          double a = 0;
          res_access[i++] = sin(a);
        }
        {
          double a = 1;
          res_access[i++] = log(a);
        }
        {
          double a = 1;
          res_access[i++] = acos(a);
        }
        {
          double a = 0;
          res_access[i++] = asin(a);
        }
        {
          double a = 0;
          res_access[i++] = atan(a);
        }
        {
          double a = 0;
          double b = 1;
          res_access[i++] = atan2(a, b);
        }
        {
          double a = 0;
          res_access[i++] = cosh(a);
        }
        {
          double a = 0;
          res_access[i++] = exp(a);
        }
        {
          double a = 1.5;
          double b = 1;
          res_access[i++] = fmod(a, b);
        }
        {
          double a = 0;
          res_access[i++] = frexp(a, &exp_access[0]);
        }
        {
          double a = 1;
          res_access[i++] = ldexp(a, 1);
        }
        {
          double a = 1;
          res_access[i++] = log10(a);
        }
        {
          double a = 1;
          res_access[i++] = modf(a, &iptr_access[0]);
        }
        {
          double a = 1;
          double b = 1;
          res_access[i++] = pow(a, b);
        }
        {
          double a = 0;
          res_access[i++] = sinh(a);
        }
        {
          double a = 4;
          res_access[i++] = sqrt(a);
        }
        {
          double a = 0;
          res_access[i++] = tan(a);
        }
        {
          double a = 0;
          res_access[i++] = tanh(a);
        }
        {
          double a = 1;
          res_access[i++] = acosh(a);
        }
        {
          double a = 0;
          res_access[i++] = asinh(a);
        }
        {
          double a = 0;
          res_access[i++] = atanh(a);
        }
        {
          double a = 1;
          res_access[i++] = cbrt(a);
        }
        {
          double a = 0;
          res_access[i++] = erf(a);
        }
        {
          double a = 0;
          res_access[i++] = erfc(a);
        }
        {
          double a = 1;
          res_access[i++] = exp2(a);
        }
        {
          double a = 0;
          res_access[i++] = expm1(a);
        }
        {
          double a = 0;
          res_access[i++] = fdim(1, a);
        }
        {
          double a = 1;
          double b = 1;
          double c = 1;
          res_access[i++] = fma(a, b, c);
        }
        {
          double a = 3;
          double b = 4;
          res_access[i++] = hypot(a, b);
        }
        {
          double a = 1;
          res_access[i++] = ilogb(a);
        }
        {
          double a = 0;
          res_access[i++] = log1p(a);
        }
        {
          double a = 1;
          res_access[i++] = log2(a);
        }
        {
          double a = 1;
          res_access[i++] = logb(a);
        }
        {
          double a = 0.5;
          double b = 1;
          res_access[i++] = remainder(a, b);
        }
        {
          double a = 0.5;
          double b = 1;
          res_access[i++] = remquo(a, b, &quo_access[0]);
        }
        {
          double a = NAN;
          res_access[i++] = tgamma(a);
        }
        {
          double a = NAN;
          res_access[i++] = lgamma(a);
        }
      });
    });
  }
  for (int i = 0; i < 36; ++i) {
    assert(is_about_FP(result[i], ref[i]));
  }
  assert(std::isnan(result[36]));
  assert(std::isnan(result[37]));
  assert(is_about_FP(iptr, b));
  assert(expv == 0);
  assert(quo == 0);
}

int main() {
  s::queue deviceQueue;
  if (deviceQueue.get_device().has_extension("cl_khr_fp64")) {
    device_math_test(deviceQueue);
    std::cout << "Pass" << std::endl;
  }
  return 0;
}
