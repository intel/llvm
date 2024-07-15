// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}
// UNSUPPORTED: gpu
// RUN: %{build} %{mathflags} -o %t.out
// RUN: %{run} %t.out

// RUN: %{build} -fsycl-device-lib-jit-link %{mathflags} -o %t.out
// RUN: %{run} %t.out

#include <array>
#include <cassert>
#include <complex>
#include <iostream>
#include <sycl/detail/core.hpp>

#include "math_utils.hpp"

using std::complex;
namespace s = sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

template <typename T> bool approx_equal_cmplx(complex<T> x, complex<T> y) {
  return approx_equal_fp(x.real(), y.real()) &&
         approx_equal_fp(x.imag(), y.imag());
}

complex<float> ref1_results[] = {
    complex<float>(-1.f, 1.f),          complex<float>(1.f, 3.f),
    complex<float>(-2.f, 10.f),         complex<float>(-8.f, 31.f),
    complex<float>(1.f, 1.f),           complex<float>(2.f, 1.f),
    complex<float>(2.f, 2.f),           complex<float>(3.f, 4.f),
    complex<float>(2.f, 1.f),           complex<float>(0.f, 1.f),
    complex<float>(2.f, 0.f),           complex<float>(0.f, 0.f),
    complex<float>(1.f, 0.f),           complex<float>(0.f, 1.f),
    complex<float>(-1.f, 0.f),          complex<float>(0.f, M_E),
    complex<float>(0.f, 0.f),           complex<float>(0.f, M_PI_2),
    complex<float>(0.f, M_PI),          complex<float>(1.f, M_PI_2),
    complex<float>(0.f, 0.f),           complex<float>(1.f, 0.f),
    complex<float>(1.f, 0.f),           complex<float>(-1.f, 0.f),
    complex<float>(-INFINITY, 0.f),     complex<float>(1.f, 0.f),
    complex<float>(10.f, 0.f),          complex<float>(100.f, 0.f),
    complex<float>(200.f, 0.f),         complex<float>(1.f, 2.f),
    complex<float>(INFINITY, 0.f),      complex<float>(INFINITY, 0.f),
    complex<float>(0.f, 1.f),           complex<float>(0.f, 0.f),
    complex<float>(1.f, 0.f),           complex<float>(INFINITY, 0.f),
    complex<float>(0.f, 0.f),           complex<float>(0.f, M_PI_2),
    complex<float>(1.f, -4.f),          complex<float>(18.f, -7.f),
    complex<float>(M_PI_2, 0.549306f),  complex<float>(INFINITY, 0.f),
    complex<float>(INFINITY, INFINITY), complex<float>(INFINITY, -INFINITY)};

float ref2_results[] = {0.f, 25.f, 169.f,    INFINITY, 0.f,
                        5.f, 13.f, INFINITY, 0.f,      M_PI_2};

complex<float> ref3_results[] = {
    complex<float>(0.f, 1.f),         complex<float>(1.f, 1.f),
    complex<float>(2.f, 0.f),         complex<float>(2.f, 3.f),
    complex<float>(M_PI_2, 0.f),      complex<float>(0.f, 0.f),
    complex<float>(1.f, 0.f),         complex<float>(0.f, 0.f),
    complex<float>(INFINITY, M_PI_2), complex<float>(INFINITY, 0.f),
    complex<float>(0.f, M_PI_2),      complex<float>(INFINITY, M_PI_2),
    complex<float>(INFINITY, 0.f),    complex<float>(1.557408f, 0.f),
    complex<float>(0.f, 0.761594f),   complex<float>(M_PI_2, 0.f),
    complex<float>(-1.f, 0.f),        complex<float>(-1.f, 0.f),
    complex<float>(-1.f, 0.f)};

static constexpr auto TestArraySize1 = std::size(ref1_results);
static constexpr auto TestArraySize2 = std::size(ref2_results);
static constexpr auto TestArraySize3 = std::size(ref3_results);

int device_complex_test_1(s::queue &deviceQueue) {
  s::range<1> numOfItems1{TestArraySize1};
  s::range<1> numOfItems2{TestArraySize2};
  std::array<complex<float>, TestArraySize1> result1;
  std::array<float, TestArraySize2> result2;
  {
    s::buffer<complex<float>, 1> buffer1(result1.data(), numOfItems1);
    s::buffer<float, 1> buffer2(result2.data(), numOfItems2);
    deviceQueue.submit([&](s::handler &cgh) {
      auto buf_out1_access = buffer1.get_access<sycl_write>(cgh);
      auto buf_out2_access = buffer2.get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceComplexTest>([=]() {
        int index = 0;
        buf_out1_access[index++] =
            complex<float>(0.f, 1.f) * complex<float>(1.f, 1.f);
        buf_out1_access[index++] =
            complex<float>(1.f, 1.f) * complex<float>(2.f, 1.f);
        buf_out1_access[index++] =
            complex<float>(2.f, 3.f) * complex<float>(2.f, 2.f);
        buf_out1_access[index++] =
            complex<float>(4.f, 5.f) * complex<float>(3.f, 4.f);
        buf_out1_access[index++] =
            complex<float>(-1.f, 1.f) / complex<float>(0.f, 1.f);
        buf_out1_access[index++] =
            complex<float>(1.f, 3.f) / complex<float>(1.f, 1.f);
        buf_out1_access[index++] =
            complex<float>(-2.f, 10.f) / complex<float>(2.f, 3.f);
        buf_out1_access[index++] =
            complex<float>(-8.f, 31.f) / complex<float>(4.f, 5.f);
        buf_out1_access[index++] =
            complex<float>(4.f, 2.f) / complex<float>(2.f, 0.f);
        buf_out1_access[index++] =
            complex<float>(-1.f, 0.f) / complex<float>(0.f, 1.f);
        buf_out1_access[index++] =
            complex<float>(0.f, 10.f) / complex<float>(0.f, 5.f);
        buf_out1_access[index++] =
            complex<float>(0.f, 0.f) / complex<float>(1.f, 0.f);
        buf_out1_access[index++] = std::exp(complex<float>(0.f, 0.f));
        buf_out1_access[index++] = std::exp(complex<float>(0.f, M_PI_2));
        buf_out1_access[index++] = std::exp(complex<float>(0.f, M_PI));
        buf_out1_access[index++] = std::exp(complex<float>(1.f, M_PI_2));
        buf_out1_access[index++] = std::log(complex<float>(1.f, 0.f));
        buf_out1_access[index++] = std::log(complex<float>(0.f, 1.f));
        buf_out1_access[index++] = std::log(complex<float>(-1.f, 0.f));
        buf_out1_access[index++] = std::log(complex<float>(0.f, M_E));
        buf_out1_access[index++] = std::sin(complex<float>(0.f, 0.f));
        buf_out1_access[index++] = std::sin(complex<float>(M_PI_2, 0.f));
        buf_out1_access[index++] = std::cos(complex<float>(0.f, 0.f));
        buf_out1_access[index++] = std::cos(complex<float>(M_PI, 0.f));
        buf_out1_access[index++] = std::log10(complex<float>(0.f, 0.f));
        buf_out1_access[index++] = std::polar(1.f);
        buf_out1_access[index++] = std::polar(10.f, 0.f);
        buf_out1_access[index++] = std::polar(100.f);
        buf_out1_access[index++] = std::polar(200.f, 0.f);
        buf_out1_access[index++] = std::proj(complex<float>(1.f, 2.f));
        buf_out1_access[index++] = std::proj(complex<float>(INFINITY, -1.f));
        buf_out1_access[index++] = std::proj(complex<float>(0.f, -INFINITY));
        buf_out1_access[index++] = std::pow(complex<float>(-1.f, 0.f), 0.5f);
        buf_out1_access[index++] = std::sinh(complex<float>(0.f, 0.f));
        buf_out1_access[index++] = std::cosh(complex<float>(0.f, 0.f));
        buf_out1_access[index++] = std::cosh(complex<float>(INFINITY, 0.f));
        buf_out1_access[index++] = std::atanh(complex<float>(0.f, 0.f));
        buf_out1_access[index++] = std::atanh(complex<float>(1.f, INFINITY));
        buf_out1_access[index++] = std::conj(complex<float>(1.f, 4.f));
        buf_out1_access[index++] = std::conj(complex<float>(18.f, 7.f));
        buf_out1_access[index++] = std::atan(complex<float>(0.f, 2.f));
        buf_out1_access[index++] = std::exp(complex<float>(1e6f, 0.f));
        buf_out1_access[index++] = std::exp(complex<float>(1e6f, 0.1f));
        buf_out1_access[index++] = std::exp(complex<float>(1e6f, -0.1f));

        index = 0;
        buf_out2_access[index++] = std::norm(complex<float>(0.f, 0.f));
        buf_out2_access[index++] = std::norm(complex<float>(3.f, 4.f));
        buf_out2_access[index++] = std::norm(complex<float>(12.f, 5.f));
        buf_out2_access[index++] = std::norm(complex<float>(INFINITY, 1.f));
        buf_out2_access[index++] = std::abs(complex<float>(0.f, 0.f));
        buf_out2_access[index++] = std::abs(complex<float>(3.f, 4.f));
        buf_out2_access[index++] = std::abs(complex<float>(12.f, 5.f));
        buf_out2_access[index++] = std::abs(complex<float>(INFINITY, 1.f));
        buf_out2_access[index++] = std::arg(complex<float>(1.f, 0.f));
        buf_out2_access[index++] = std::arg(complex<float>(0.f, 1.f));
      });
    });
  }

  int n_fails = 0;
  for (size_t idx = 0; idx < TestArraySize1; ++idx) {
    if (!approx_equal_cmplx(result1[idx], ref1_results[idx])) {
      ++n_fails;
      std::cout << "test array 1 fail at index " << idx << "\n";
      std::cout << "expected: " << ref1_results[idx] << "\n";
      std::cout << "actual:   " << result1[idx] << "\n";
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

// The MSVC implementation of some complex<float> math functions depends on
// some 'double' C math functions such as ldexp, those complex<float> math
// functions can only work on Windows with fp64 extension support from
// underlying device.
#ifndef _WIN32
int device_complex_test_2(s::queue &deviceQueue) {
  s::range<1> numOfItems1{TestArraySize3};
  std::array<complex<float>, TestArraySize3> result3;
  {
    s::buffer<complex<float>, 1> buffer1(result3.data(), numOfItems1);
    deviceQueue.submit([&](s::handler &cgh) {
      auto buf_out1_access = buffer1.get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceComplexTest2>([=]() {
        int index = 0;
        buf_out1_access[index++] = std::sqrt(complex<float>(-1.f, 0.f));
        buf_out1_access[index++] = std::sqrt(complex<float>(0.f, 2.f));
        buf_out1_access[index++] = std::sqrt(complex<float>(4.f, 0.f));
        buf_out1_access[index++] = std::sqrt(complex<float>(-5.f, 12.f));
        buf_out1_access[index++] = std::acos(complex<float>(0.f, 0.f));
        buf_out1_access[index++] = std::tanh(complex<float>(0.f, 0.f));
        buf_out1_access[index++] = std::tanh(complex<float>(INFINITY, 1.f));
        buf_out1_access[index++] = std::asinh(complex<float>(0.f, 0.f));
        buf_out1_access[index++] = std::asinh(complex<float>(1.f, INFINITY));
        buf_out1_access[index++] = std::asinh(complex<float>(INFINITY, 1.f));
        buf_out1_access[index++] = std::acosh(complex<float>(0.f, 0.f));
        buf_out1_access[index++] = std::acosh(complex<float>(1.f, INFINITY));
        buf_out1_access[index++] = std::acosh(complex<float>(INFINITY, 1.f));
        buf_out1_access[index++] = std::tan(complex<float>(1.f, 0.f));
        buf_out1_access[index++] = std::tan(complex<float>(0.f, 1.f));
        buf_out1_access[index++] = std::asin(complex<float>(1.f, 0.f));
        buf_out1_access[index++] = std::tanh(complex<float>(-INFINITY, NAN));
        buf_out1_access[index++] =
            std::tanh(complex<float>(-INFINITY, -INFINITY));
        buf_out1_access[index++] = std::tanh(complex<float>(-INFINITY, -2.f));
      });
    });
  }

  int n_fails = 0;
  for (size_t idx = 0; idx < TestArraySize3; ++idx) {
    if (!approx_equal_cmplx(result3[idx], ref3_results[idx])) {
      ++n_fails;
      std::cout << "test array 3 fail at index " << idx << "\n";
      std::cout << "expected: " << ref3_results[idx] << "\n";
      std::cout << "actual:   " << result3[idx] << "\n";
    }
  }
  return n_fails;
}
#endif
int main() {
  s::queue deviceQueue;

#ifdef _WIN32
  // std::complex math on Windows uses doubles internally so fp64 is required to
  // run this test.
  if (!deviceQueue.get_device().has(s::aspect::fp64)) {
    std::cout << "Skipping test as device does not support fp64 which is "
                 "required for math operations on std::complex on Windows."
              << std::endl;
    return 0;
  }
#endif

  int n_fails = 0;
  n_fails += device_complex_test_1(deviceQueue);
#ifndef _WIN32
  n_fails += device_complex_test_2(deviceQueue);
#endif
  if (n_fails == 0)
    std::cout << "Pass" << std::endl;
  return n_fails;
}
