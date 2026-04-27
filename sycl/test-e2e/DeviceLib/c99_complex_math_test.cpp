// REQUIRES: windows
// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}
// RUN: %{build} %{mathflags} -o %t1.out
// RUN: %{run} %t1.out

// This test is a C99-complex version of std_complex_math_test.cpp.
// It uses _Complex float (C99 complex type) and C99 complex math functions
// (from complex.h) instead of std::complex<float>.
// Unlike the std::complex version, the C99 complex device library
// implementations do not depend on double precision, so all tests
// run on both Linux and Windows without requiring fp64.

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

// C99 complex construction macro.
// Uses compound literal syntax supported by GCC and clang in C++ mode.
#define CMPLXF(r, i) ((float _Complex){(float)(r), (float)(i)})

// Declare C99 complex functions.
// On Linux, these are provided by libm.
// On Windows, these are provided by the Universal CRT.
// On device, the SYCL device library provides implementations.
extern "C" {
float crealf(float _Complex z);
float cimagf(float _Complex z);
float cabsf(float _Complex z);
float cargf(float _Complex z);
float _Complex cprojf(float _Complex z);
float _Complex conjf(float _Complex z);
float _Complex cexpf(float _Complex z);
float _Complex clogf(float _Complex z);
float _Complex cpowf(float _Complex x, float _Complex y);
float _Complex csqrtf(float _Complex z);
float _Complex csinf(float _Complex z);
float _Complex ccosf(float _Complex z);
float _Complex ctanf(float _Complex z);
float _Complex casinf(float _Complex z);
float _Complex cacosf(float _Complex z);
float _Complex catanf(float _Complex z);
float _Complex csinhf(float _Complex z);
float _Complex ccoshf(float _Complex z);
float _Complex ctanhf(float _Complex z);
float _Complex casinhf(float _Complex z);
float _Complex cacoshf(float _Complex z);
float _Complex catanhf(float _Complex z);
}

namespace s = sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

bool approx_equal_cmplx_f(float _Complex x, float _Complex y) {
  return approx_equal_fp(__real__ x, __real__ y) &&
         approx_equal_fp(__imag__ x, __imag__ y);
}

static float _Complex ref1_results[] = {
    CMPLXF(-1.f, 1.f),          CMPLXF(1.f, 3.f),
    CMPLXF(-2.f, 10.f),         CMPLXF(-8.f, 31.f),
    CMPLXF(1.f, 1.f),           CMPLXF(2.f, 1.f),
    CMPLXF(2.f, 2.f),           CMPLXF(3.f, 4.f),
    CMPLXF(2.f, 1.f),           CMPLXF(0.f, 1.f),
    CMPLXF(2.f, 0.f),           CMPLXF(0.f, 0.f),
    CMPLXF(1.f, 0.f),           CMPLXF(0.f, 1.f),
    CMPLXF(-1.f, 0.f),          CMPLXF(0.f, M_E),
    CMPLXF(0.f, 0.f),           CMPLXF(0.f, M_PI_2),
    CMPLXF(0.f, M_PI),          CMPLXF(1.f, M_PI_2),
    CMPLXF(0.f, 0.f),           CMPLXF(1.f, 0.f),
    CMPLXF(1.f, 0.f),           CMPLXF(-1.f, 0.f),
    CMPLXF(-INFINITY, 0.f),     CMPLXF(1.f, 0.f),
    CMPLXF(10.f, 0.f),          CMPLXF(100.f, 0.f),
    CMPLXF(200.f, 0.f),         CMPLXF(1.f, 2.f),
    CMPLXF(INFINITY, 0.f),      CMPLXF(INFINITY, 0.f),
    CMPLXF(0.f, 1.f),           CMPLXF(0.f, 0.f),
    CMPLXF(1.f, 0.f),           CMPLXF(INFINITY, 0.f),
    CMPLXF(0.f, 0.f),           CMPLXF(0.f, M_PI_2),
    CMPLXF(1.f, -4.f),          CMPLXF(18.f, -7.f),
    CMPLXF(M_PI_2, 0.549306f),  CMPLXF(INFINITY, 0.f),
    CMPLXF(INFINITY, INFINITY), CMPLXF(INFINITY, -INFINITY)};

static float ref2_results[] = {0.f, 5.f, 13.f, INFINITY, 0.f, M_PI_2};

static float _Complex ref3_results[] = {
    CMPLXF(0.f, 1.f),         CMPLXF(1.f, 1.f),
    CMPLXF(2.f, 0.f),         CMPLXF(2.f, 3.f),
    CMPLXF(M_PI_2, 0.f),      CMPLXF(0.f, 0.f),
    CMPLXF(1.f, 0.f),         CMPLXF(0.f, 0.f),
    CMPLXF(INFINITY, M_PI_2), CMPLXF(INFINITY, 0.f),
    CMPLXF(0.f, M_PI_2),      CMPLXF(INFINITY, M_PI_2),
    CMPLXF(INFINITY, 0.f),    CMPLXF(1.557408f, 0.f),
    CMPLXF(0.f, 0.761594f),   CMPLXF(M_PI_2, 0.f),
    CMPLXF(-1.f, 0.f),        CMPLXF(-1.f, 0.f),
    CMPLXF(-1.f, 0.f)};

static constexpr auto TestArraySize1 = sizeof(ref1_results) / sizeof(ref1_results[0]);
static constexpr auto TestArraySize2 = sizeof(ref2_results) / sizeof(ref2_results[0]);
static constexpr auto TestArraySize3 = sizeof(ref3_results) / sizeof(ref3_results[0]);

int device_complex_test_1(s::queue &deviceQueue) {
  s::range<1> numOfItems1{TestArraySize1};
  s::range<1> numOfItems2{TestArraySize2};
  std::array<float _Complex, TestArraySize1> result1;
  std::array<float, TestArraySize2> result2;
  {
    s::buffer<float _Complex, 1> buffer1(result1.data(), numOfItems1);
    s::buffer<float, 1> buffer2(result2.data(), numOfItems2);
    deviceQueue.submit([&](s::handler &cgh) {
      auto buf_out1_access = buffer1.get_access<sycl_write>(cgh);
      auto buf_out2_access = buffer2.get_access<sycl_write>(cgh);
      cgh.single_task<class C99DeviceComplexTest>([=]() {
        int index = 0;
        buf_out1_access[index++] = CMPLXF(0.f, 1.f) * CMPLXF(1.f, 1.f);
        buf_out1_access[index++] = CMPLXF(1.f, 1.f) * CMPLXF(2.f, 1.f);
        buf_out1_access[index++] = CMPLXF(2.f, 3.f) * CMPLXF(2.f, 2.f);
        buf_out1_access[index++] = CMPLXF(4.f, 5.f) * CMPLXF(3.f, 4.f);
        buf_out1_access[index++] = CMPLXF(-1.f, 1.f) / CMPLXF(0.f, 1.f);
        buf_out1_access[index++] = CMPLXF(1.f, 3.f) / CMPLXF(1.f, 1.f);
        buf_out1_access[index++] = CMPLXF(-2.f, 10.f) / CMPLXF(2.f, 3.f);
        buf_out1_access[index++] = CMPLXF(-8.f, 31.f) / CMPLXF(4.f, 5.f);
        buf_out1_access[index++] = CMPLXF(4.f, 2.f) / CMPLXF(2.f, 0.f);
        buf_out1_access[index++] = CMPLXF(-1.f, 0.f) / CMPLXF(0.f, 1.f);
        buf_out1_access[index++] = CMPLXF(0.f, 10.f) / CMPLXF(0.f, 5.f);
        buf_out1_access[index++] = CMPLXF(0.f, 0.f) / CMPLXF(1.f, 0.f);
        buf_out1_access[index++] = cexpf(CMPLXF(0.f, 0.f));
        buf_out1_access[index++] = cexpf(CMPLXF(0.f, M_PI_2));
        buf_out1_access[index++] = cexpf(CMPLXF(0.f, M_PI));
        buf_out1_access[index++] = cexpf(CMPLXF(1.f, M_PI_2));
        buf_out1_access[index++] = clogf(CMPLXF(1.f, 0.f));
        buf_out1_access[index++] = clogf(CMPLXF(0.f, 1.f));
        buf_out1_access[index++] = clogf(CMPLXF(-1.f, 0.f));
        buf_out1_access[index++] = clogf(CMPLXF(0.f, M_E));
        buf_out1_access[index++] = csinf(CMPLXF(0.f, 0.f));
        buf_out1_access[index++] = csinf(CMPLXF(M_PI_2, 0.f));
        buf_out1_access[index++] = ccosf(CMPLXF(0.f, 0.f));
        buf_out1_access[index++] = ccosf(CMPLXF(M_PI, 0.f));
        // clog10 is not in C99; compute as clogf(z) / logf(10).
        buf_out1_access[index++] = clogf(CMPLXF(0.f, 0.f)) / logf(10.0f);
        // cpolar is not in C99; compute polar form inline.
        buf_out1_access[index++] = CMPLXF(1.f * cosf(0.f), 1.f * sinf(0.f));
        buf_out1_access[index++] = CMPLXF(10.f * cosf(0.f), 10.f * sinf(0.f));
        buf_out1_access[index++] =
            CMPLXF(100.f * cosf(0.f), 100.f * sinf(0.f));
        buf_out1_access[index++] =
            CMPLXF(200.f * cosf(0.f), 200.f * sinf(0.f));
        buf_out1_access[index++] = cprojf(CMPLXF(1.f, 2.f));
        buf_out1_access[index++] = cprojf(CMPLXF(INFINITY, -1.f));
        buf_out1_access[index++] = cprojf(CMPLXF(0.f, -INFINITY));
        buf_out1_access[index++] = cpowf(CMPLXF(-1.f, 0.f), CMPLXF(0.5f, 0.f));
        buf_out1_access[index++] = csinhf(CMPLXF(0.f, 0.f));
        buf_out1_access[index++] = ccoshf(CMPLXF(0.f, 0.f));
        buf_out1_access[index++] = ccoshf(CMPLXF(INFINITY, 0.f));
        buf_out1_access[index++] = catanhf(CMPLXF(0.f, 0.f));
        buf_out1_access[index++] = catanhf(CMPLXF(1.f, INFINITY));
        buf_out1_access[index++] = conjf(CMPLXF(1.f, 4.f));
        buf_out1_access[index++] = conjf(CMPLXF(18.f, 7.f));
        buf_out1_access[index++] = catanf(CMPLXF(0.f, 2.f));
        buf_out1_access[index++] = cexpf(CMPLXF(1e6f, 0.f));
        buf_out1_access[index++] = cexpf(CMPLXF(1e6f, 0.1f));
        buf_out1_access[index++] = cexpf(CMPLXF(1e6f, -0.1f));

        index = 0;
        buf_out2_access[index++] = cabsf(CMPLXF(0.f, 0.f));
        buf_out2_access[index++] = cabsf(CMPLXF(3.f, 4.f));
        buf_out2_access[index++] = cabsf(CMPLXF(12.f, 5.f));
        buf_out2_access[index++] = cabsf(CMPLXF(INFINITY, 1.f));
        buf_out2_access[index++] = cargf(CMPLXF(1.f, 0.f));
        buf_out2_access[index++] = cargf(CMPLXF(0.f, 1.f));
      });
    });
  }

  int n_fails = 0;
  for (size_t idx = 0; idx < TestArraySize1; ++idx) {
    if (!approx_equal_cmplx_f(result1[idx], ref1_results[idx])) {
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

// Unlike the std::complex version, C99 complex float math functions in the
// SYCL device library do not depend on double precision, so this test
// runs on both Linux and Windows.
int device_complex_test_2(s::queue &deviceQueue) {
  s::range<1> numOfItems1{TestArraySize3};
  std::array<float _Complex, TestArraySize3> result3;
  {
    s::buffer<float _Complex, 1> buffer1(result3.data(), numOfItems1);
    deviceQueue.submit([&](s::handler &cgh) {
      auto buf_out1_access = buffer1.get_access<sycl_write>(cgh);
      cgh.single_task<class C99DeviceComplexTest2>([=]() {
        int index = 0;
        buf_out1_access[index++] = csqrtf(CMPLXF(-1.f, 0.f));
        buf_out1_access[index++] = csqrtf(CMPLXF(0.f, 2.f));
        buf_out1_access[index++] = csqrtf(CMPLXF(4.f, 0.f));
        buf_out1_access[index++] = csqrtf(CMPLXF(-5.f, 12.f));
        buf_out1_access[index++] = cacosf(CMPLXF(0.f, 0.f));
        buf_out1_access[index++] = ctanhf(CMPLXF(0.f, 0.f));
        buf_out1_access[index++] = ctanhf(CMPLXF(INFINITY, 1.f));
        buf_out1_access[index++] = casinhf(CMPLXF(0.f, 0.f));
        buf_out1_access[index++] = casinhf(CMPLXF(1.f, INFINITY));
        buf_out1_access[index++] = casinhf(CMPLXF(INFINITY, 1.f));
        buf_out1_access[index++] = cacoshf(CMPLXF(0.f, 0.f));
        buf_out1_access[index++] = cacoshf(CMPLXF(1.f, INFINITY));
        buf_out1_access[index++] = cacoshf(CMPLXF(INFINITY, 1.f));
        buf_out1_access[index++] = ctanf(CMPLXF(1.f, 0.f));
        buf_out1_access[index++] = ctanf(CMPLXF(0.f, 1.f));
        buf_out1_access[index++] = casinf(CMPLXF(1.f, 0.f));
        buf_out1_access[index++] = ctanhf(CMPLXF(-INFINITY, NAN));
        buf_out1_access[index++] = ctanhf(CMPLXF(-INFINITY, -INFINITY));
        buf_out1_access[index++] = ctanhf(CMPLXF(-INFINITY, -2.f));
      });
    });
  }

  int n_fails = 0;
  for (size_t idx = 0; idx < TestArraySize3; ++idx) {
    if (!approx_equal_cmplx_f(result3[idx], ref3_results[idx])) {
      ++n_fails;
      std::cout << "test array 3 fail at index " << idx << "\n";
      std::cout << "expected: (" << __real__ ref3_results[idx] << ", "
                << __imag__ ref3_results[idx] << ")\n";
      std::cout << "actual:   (" << __real__ result3[idx] << ", "
                << __imag__ result3[idx] << ")\n";
    }
  }
  return n_fails;
}

int main() {
  s::queue deviceQueue;

  int n_fails = 0;
  n_fails += device_complex_test_1(deviceQueue);
  // C99 complex device library implementations do not depend on double
  // precision, so all tests run on both Linux and Windows.
  n_fails += device_complex_test_2(deviceQueue);
  if (n_fails == 0)
    std::cout << "Pass" << std::endl;
  return n_fails;
}
