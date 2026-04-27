// REQUIRES: aspect-fp64, windows
// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}
// RUN: %{build} %{mathflags} -o %t1.out
// RUN: %{run} %t1.out

// This test is a C99-complex version of std_complex_math_fp64_test.cpp.
// It uses _Complex double (C99 complex type) and C99 complex math functions
// (from complex.h) instead of std::complex<double>.
// Unlike the std::complex version, the C99 complex device library
// implementations handle tanh(-inf + nan*i) and tanh(-inf + -inf*i) correctly
// on both Linux and Windows.

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#include <sycl/detail/core.hpp>

// _USE_MATH_DEFINES must be defined in order to use math constants in MSVC.
#ifdef _WIN32
#define _USE_MATH_DEFINES 1
#include <math.h>
#endif

#include "math_utils.hpp"

// C99 complex construction macros.
// Uses compound literal syntax supported by GCC and clang in C++ mode.
#define CMPLX(r, i) ((double _Complex){(double)(r), (double)(i)})

// Declare C99 complex functions (double precision).
// On Linux, these are provided by libm.
// On Windows, these are provided by the Universal CRT.
// On device, the SYCL device library provides implementations.
extern "C" {
double creal(double _Complex z);
double cimag(double _Complex z);
double cabs(double _Complex z);
double carg(double _Complex z);
double _Complex cproj(double _Complex z);
double _Complex conj(double _Complex z);
double _Complex cexp(double _Complex z);
double _Complex clog(double _Complex z);
double _Complex cpow(double _Complex x, double _Complex y);
double _Complex csqrt(double _Complex z);
double _Complex csin(double _Complex z);
double _Complex ccos(double _Complex z);
double _Complex ctan(double _Complex z);
double _Complex casin(double _Complex z);
double _Complex cacos(double _Complex z);
double _Complex catan(double _Complex z);
double _Complex csinh(double _Complex z);
double _Complex ccosh(double _Complex z);
double _Complex ctanh(double _Complex z);
double _Complex casinh(double _Complex z);
double _Complex cacosh(double _Complex z);
double _Complex catanh(double _Complex z);
}

namespace s = sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

bool approx_equal_cmplx_d(double _Complex x, double _Complex y) {
  return approx_equal_fp(__real__ x, __real__ y) &&
         approx_equal_fp(__imag__ x, __imag__ y);
}

// Unlike the std::complex version, the C99 complex device library
// handles tanh(-inf + nan*i) and tanh(-inf + -inf*i) correctly on all
// platforms, so no _WIN32 workaround is needed in the reference data.
static double _Complex ref1_results[] = {
    CMPLX(-1., 1.),
    CMPLX(1., 3.),
    CMPLX(-2., 10.),
    CMPLX(-8., 31.),
    CMPLX(1., 1.),
    CMPLX(2., 1.),
    CMPLX(2., 2.),
    CMPLX(3., 4.),
    CMPLX(2., 1.),
    CMPLX(0., 1.),
    CMPLX(2., 0.),
    CMPLX(0., 0.),
    CMPLX(0., 1.),
    CMPLX(1., 1.),
    CMPLX(2., 0.),
    CMPLX(2., 3.),
    CMPLX(1., 0.),
    CMPLX(0., 1.),
    CMPLX(-1., 0.),
    CMPLX(0., M_E),
    CMPLX(0., 0.),
    CMPLX(0., M_PI_2),
    CMPLX(0., M_PI),
    CMPLX(1., M_PI_2),
    CMPLX(0., 0.),
    CMPLX(1., 0.),
    CMPLX(1., 0.),
    CMPLX(-1., 0.),
    CMPLX(-INFINITY, 0.),
    CMPLX(1., 0.),
    CMPLX(10., 0.),
    CMPLX(100., 0.),
    CMPLX(200., 0.),
    CMPLX(1., 2.),
    CMPLX(INFINITY, 0.),
    CMPLX(INFINITY, 0.),
    CMPLX(0., 1.),
    CMPLX(M_PI_2, 0.),
    CMPLX(0., 0.),
    CMPLX(1., 0.),
    CMPLX(INFINITY, 0.),
    CMPLX(0., 0.),
    CMPLX(1., 0.),
    CMPLX(0., 0.),
    CMPLX(INFINITY, M_PI_2),
    CMPLX(INFINITY, 0.),
    CMPLX(0., M_PI_2),
    CMPLX(INFINITY, M_PI_2),
    CMPLX(INFINITY, 0.),
    CMPLX(0., 0.),
    CMPLX(0., M_PI_2),

    CMPLX(1., -4.),
    CMPLX(18., -7.),
    CMPLX(1.557407724654902, 0.),
    CMPLX(0, 0.761594155955765),
    CMPLX(M_PI_2, 0.),
    CMPLX(M_PI_2, 0.549306144334055),
    CMPLX(-1., 0.),
    CMPLX(-1., 0.),
    CMPLX(-1., 0.),
    CMPLX(INFINITY, 0.),
    CMPLX(INFINITY, INFINITY),
    CMPLX(INFINITY, -INFINITY)};

static double ref2_results[] = {0., 5., 13., INFINITY, 0., M_PI_2};

static constexpr auto TestArraySize1 = sizeof(ref1_results) / sizeof(ref1_results[0]);
static constexpr auto TestArraySize2 = 6;

int device_complex_test(s::queue &deviceQueue) {
  s::range<1> numOfItems1{TestArraySize1};
  s::range<1> numOfItems2{TestArraySize2};
  std::array<double _Complex, TestArraySize1> result1;
  std::array<double, TestArraySize2> result2;
  {
    s::buffer<double _Complex, 1> buffer1(result1.data(), numOfItems1);
    s::buffer<double, 1> buffer2(result2.data(), numOfItems2);
    deviceQueue.submit([&](s::handler &cgh) {
      auto buf_out1_access = buffer1.get_access<sycl_write>(cgh);
      auto buf_out2_access = buffer2.get_access<sycl_write>(cgh);
      cgh.single_task<class C99DeviceComplexFP64Test>([=]() {
        int index = 0;
        buf_out1_access[index++] = CMPLX(0., 1.) * CMPLX(1., 1.);
        buf_out1_access[index++] = CMPLX(1., 1.) * CMPLX(2., 1.);
        buf_out1_access[index++] = CMPLX(2., 3.) * CMPLX(2., 2.);
        buf_out1_access[index++] = CMPLX(4., 5.) * CMPLX(3., 4.);
        buf_out1_access[index++] = CMPLX(-1., 1.) / CMPLX(0., 1.);
        buf_out1_access[index++] = CMPLX(1., 3.) / CMPLX(1., 1.);
        buf_out1_access[index++] = CMPLX(-2., 10.) / CMPLX(2., 3.);
        buf_out1_access[index++] = CMPLX(-8., 31.) / CMPLX(4., 5.);
        buf_out1_access[index++] = CMPLX(4., 2.) / CMPLX(2., 0.);
        buf_out1_access[index++] = CMPLX(-1., 0.) / CMPLX(0., 1.);
        buf_out1_access[index++] = CMPLX(0., 10.) / CMPLX(0., 5.);
        buf_out1_access[index++] = CMPLX(0., 0.) / CMPLX(1., 0.);
        buf_out1_access[index++] = csqrt(CMPLX(-1., 0.));
        buf_out1_access[index++] = csqrt(CMPLX(0., 2.));
        buf_out1_access[index++] = csqrt(CMPLX(4., 0.));
        buf_out1_access[index++] = csqrt(CMPLX(-5., 12.));
        buf_out1_access[index++] = cexp(CMPLX(0., 0.));
        buf_out1_access[index++] = cexp(CMPLX(0., M_PI_2));
        buf_out1_access[index++] = cexp(CMPLX(0., M_PI));
        buf_out1_access[index++] = cexp(CMPLX(1., M_PI_2));
        buf_out1_access[index++] = clog(CMPLX(1., 0.));
        buf_out1_access[index++] = clog(CMPLX(0., 1.));
        buf_out1_access[index++] = clog(CMPLX(-1., 0.));
        buf_out1_access[index++] = clog(CMPLX(0., M_E));
        buf_out1_access[index++] = csin(CMPLX(0., 0.));
        buf_out1_access[index++] = csin(CMPLX(M_PI_2, 0.));
        buf_out1_access[index++] = ccos(CMPLX(0., 0.));
        buf_out1_access[index++] = ccos(CMPLX(M_PI, 0.));
        // clog10 is not in C99; compute as clog(z) / log(10).
        buf_out1_access[index++] = clog(CMPLX(0., 0.)) / log(10.0);
        // cpolar is not in C99; compute polar form inline.
        buf_out1_access[index++] = CMPLX(1. * cos(0.), 1. * sin(0.));
        buf_out1_access[index++] = CMPLX(10. * cos(0.), 10. * sin(0.));
        buf_out1_access[index++] = CMPLX(100. * cos(0.), 100. * sin(0.));
        buf_out1_access[index++] = CMPLX(200. * cos(0.), 200. * sin(0.));
        buf_out1_access[index++] = cproj(CMPLX(1., 2.));
        buf_out1_access[index++] = cproj(CMPLX(INFINITY, -1.));
        buf_out1_access[index++] = cproj(CMPLX(0., -INFINITY));
        buf_out1_access[index++] = cpow(CMPLX(-1., 0.), CMPLX(0.5, 0.));
        buf_out1_access[index++] = cacos(CMPLX(0., 0.));
        buf_out1_access[index++] = csinh(CMPLX(0., 0.));
        buf_out1_access[index++] = ccosh(CMPLX(0., 0.));
        buf_out1_access[index++] = ccosh(CMPLX(INFINITY, 0.));
        buf_out1_access[index++] = ctanh(CMPLX(0., 0.));
        buf_out1_access[index++] = ctanh(CMPLX(INFINITY, 1.));
        buf_out1_access[index++] = casinh(CMPLX(0., 0.));
        buf_out1_access[index++] = casinh(CMPLX(1., INFINITY));
        buf_out1_access[index++] = casinh(CMPLX(INFINITY, 1.));
        buf_out1_access[index++] = cacosh(CMPLX(0., 0.));
        buf_out1_access[index++] = cacosh(CMPLX(1., INFINITY));
        buf_out1_access[index++] = cacosh(CMPLX(INFINITY, 1.));
        buf_out1_access[index++] = catanh(CMPLX(0., 0.));
        buf_out1_access[index++] = catanh(CMPLX(1., INFINITY));
        buf_out1_access[index++] = conj(CMPLX(1., 4.));
        buf_out1_access[index++] = conj(CMPLX(18., 7.));
        buf_out1_access[index++] = ctan(CMPLX(1., 0.));
        buf_out1_access[index++] = ctan(CMPLX(0., 1.));
        buf_out1_access[index++] = casin(CMPLX(1., 0.));
        buf_out1_access[index++] = catan(CMPLX(0., 2.));
        // C99 complex device library handles tanh correctly on all platforms.
        buf_out1_access[index++] = ctanh(CMPLX(-INFINITY, NAN));
        buf_out1_access[index++] = ctanh(CMPLX(-INFINITY, -INFINITY));
        buf_out1_access[index++] = ctanh(CMPLX(-INFINITY, -2.));
        buf_out1_access[index++] = cexp(CMPLX(1e6, 0.));
        buf_out1_access[index++] = cexp(CMPLX(1e6, 0.1));
        buf_out1_access[index++] = cexp(CMPLX(1e6, -0.1));

        index = 0;
        buf_out2_access[index++] = cabs(CMPLX(0., 0.));
        buf_out2_access[index++] = cabs(CMPLX(3., 4.));
        buf_out2_access[index++] = cabs(CMPLX(12., 5.));
        buf_out2_access[index++] = cabs(CMPLX(INFINITY, 1.));
        buf_out2_access[index++] = carg(CMPLX(1., 0.));
        buf_out2_access[index++] = carg(CMPLX(0., 1.));
      });
    });
  }

  int n_fails = 0;
  for (size_t idx = 0; idx < TestArraySize1; ++idx) {
    if (!approx_equal_cmplx_d(result1[idx], ref1_results[idx])) {
      ++n_fails;
      std::cout << "test array 1 fail at index " << idx << "\n";
      std::cout << "expected: (" << __real__ ref1_results[idx] << ", "
                << __imag__ ref1_results[idx] << ")\n";
      std::cout << "actual:   (" << __real__ result1[idx] << ", "
                << __imag__ result1[idx] << ")\n";
    }
  }
  for (size_t idx = 0; idx < TestArraySize2; ++idx) {
    if (!approx_equal_fp(result2[idx], ref2_results[idx])) {
      ++n_fails;
      std::cout << "test array 2 fail at index " << idx << "\n";
      std::cout << "expected: " << ref2_results[idx] << "\n";
      std::cout << "actual:   " << result2[idx] << "\n";
    }
  }
  return n_fails;
}

int main() {
  s::queue deviceQueue;
  auto n_fails = device_complex_test(deviceQueue);
  if (n_fails == 0)
    std::cout << "Pass" << std::endl;
  return n_fails;
}
