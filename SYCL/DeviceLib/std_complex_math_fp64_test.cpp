// RUN: %clangxx -fsycl  %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// RUN: %clangxx -fsycl -fsycl-device-lib-jit-link %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <array>
#include <cassert>
#include <complex>

#include "math_utils.hpp"

using std::complex;
namespace s = cl::sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

template <typename T> bool approx_equal_cmplx(complex<T> x, complex<T> y) {
  return approx_equal_fp(x.real(), y.real()) &&
         approx_equal_fp(x.imag(), y.imag());
}

static constexpr auto TestArraySize1 = 57;
static constexpr auto TestArraySize2 = 10;

std::array<complex<double>, TestArraySize1> ref1_results = {
    complex<double>(-1., 1.),
    complex<double>(1., 3.),
    complex<double>(-2., 10.),
    complex<double>(-8., 31.),
    complex<double>(1., 1.),
    complex<double>(2., 1.),
    complex<double>(2., 2.),
    complex<double>(3., 4.),
    complex<double>(2., 1.),
    complex<double>(0., 1.),
    complex<double>(2., 0.),
    complex<double>(0., 0.),
    complex<double>(0., 1.),
    complex<double>(1., 1.),
    complex<double>(2., 0.),
    complex<double>(2., 3.),
    complex<double>(1., 0.),
    complex<double>(0., 1.),
    complex<double>(-1., 0.),
    complex<double>(0., M_E),
    complex<double>(0., 0.),
    complex<double>(0., M_PI_2),
    complex<double>(0., M_PI),
    complex<double>(1., M_PI_2),
    complex<double>(0., 0.),
    complex<double>(1., 0.),
    complex<double>(1., 0.),
    complex<double>(-1., 0.),
    complex<double>(-INFINITY, 0.),
    complex<double>(1., 0.),
    complex<double>(10., 0.),
    complex<double>(100., 0.),
    complex<double>(200., 0.),
    complex<double>(1., 2.),
    complex<double>(INFINITY, 0.),
    complex<double>(INFINITY, 0.),
    complex<double>(0., 1.),
    complex<double>(M_PI_2, 0.),
    complex<double>(0., 0.),
    complex<double>(1., 0.),
    complex<double>(INFINITY, 0.),
    complex<double>(0., 0.),
    complex<double>(1., 0.),
    complex<double>(0., 0.),
    complex<double>(INFINITY, M_PI_2),
    complex<double>(INFINITY, 0.),
    complex<double>(0., M_PI_2),
    complex<double>(INFINITY, M_PI_2),
    complex<double>(INFINITY, 0.),
    complex<double>(0., 0.),
    complex<double>(0., M_PI_2),

    complex<double>(1., -4.),
    complex<double>(18., -7.),
    complex<double>(1.557407724654902, 0.),
    complex<double>(0, 0.761594155955765),
    complex<double>(M_PI_2, 0.),
    complex<double>(M_PI_2, 0.549306144334055)};

std::array<double, TestArraySize2> ref2_results = {
    0., 25., 169., INFINITY, 0., 5., 13., INFINITY, 0., M_PI_2};

void device_complex_test(s::queue &deviceQueue) {
  s::range<1> numOfItems1{TestArraySize1};
  s::range<1> numOfItems2{TestArraySize2};
  std::array<complex<double>, TestArraySize1> result1;
  std::array<double, TestArraySize2> result2;
  {
    s::buffer<complex<double>, 1> buffer1(result1.data(), numOfItems1);
    s::buffer<double, 1> buffer2(result2.data(), numOfItems2);
    deviceQueue.submit([&](s::handler &cgh) {
      auto buf_out1_access = buffer1.get_access<sycl_write>(cgh);
      auto buf_out2_access = buffer2.get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceComplexTest>([=]() {
        int index = 0;
        buf_out1_access[index++] =
            complex<double>(0., 1.) * complex<double>(1., 1.);
        buf_out1_access[index++] =
            complex<double>(1., 1.) * complex<double>(2., 1.);
        buf_out1_access[index++] =
            complex<double>(2., 3.) * complex<double>(2., 2.);
        buf_out1_access[index++] =
            complex<double>(4., 5.) * complex<double>(3., 4.);
        buf_out1_access[index++] =
            complex<double>(-1., 1.) / complex<double>(0., 1.);
        buf_out1_access[index++] =
            complex<double>(1., 3.) / complex<double>(1., 1.);
        buf_out1_access[index++] =
            complex<double>(-2., 10.) / complex<double>(2., 3.);
        buf_out1_access[index++] =
            complex<double>(-8., 31.) / complex<double>(4., 5.);
        buf_out1_access[index++] =
            complex<double>(4., 2.) / complex<double>(2., 0.);
        buf_out1_access[index++] =
            complex<double>(-1., 0.) / complex<double>(0., 1.);
        buf_out1_access[index++] =
            complex<double>(0., 10.) / complex<double>(0., 5.);
        buf_out1_access[index++] =
            complex<double>(0., 0.) / complex<double>(1., 0.);
        buf_out1_access[index++] = std::sqrt(complex<double>(-1., 0.));
        buf_out1_access[index++] = std::sqrt(complex<double>(0., 2.));
        buf_out1_access[index++] = std::sqrt(complex<double>(4., 0.));
        buf_out1_access[index++] = std::sqrt(complex<double>(-5., 12.));
        buf_out1_access[index++] = std::exp(complex<double>(0., 0.));
        buf_out1_access[index++] = std::exp(complex<double>(0., M_PI_2));
        buf_out1_access[index++] = std::exp(complex<double>(0., M_PI));
        buf_out1_access[index++] = std::exp(complex<double>(1., M_PI_2));
        buf_out1_access[index++] = std::log(complex<double>(1., 0.));
        buf_out1_access[index++] = std::log(complex<double>(0., 1.));
        buf_out1_access[index++] = std::log(complex<double>(-1., 0.));
        buf_out1_access[index++] = std::log(complex<double>(0., M_E));
        buf_out1_access[index++] = std::sin(complex<double>(0., 0.));
        buf_out1_access[index++] = std::sin(complex<double>(M_PI_2, 0.));
        buf_out1_access[index++] = std::cos(complex<double>(0., 0.));
        buf_out1_access[index++] = std::cos(complex<double>(M_PI, 0.));
        buf_out1_access[index++] = std::log10(complex<double>(0., 0.));
        buf_out1_access[index++] = std::polar(1.);
        buf_out1_access[index++] = std::polar(10., 0.);
        buf_out1_access[index++] = std::polar(100.);
        buf_out1_access[index++] = std::polar(200., 0.);
        buf_out1_access[index++] = std::proj(complex<double>(1., 2.));
        buf_out1_access[index++] = std::proj(complex<double>(INFINITY, -1.));
        buf_out1_access[index++] = std::proj(complex<double>(0., -INFINITY));
        buf_out1_access[index++] = std::pow(complex<double>(-1., 0.), 0.5);
        buf_out1_access[index++] = std::acos(complex<double>(0., 0.));
        buf_out1_access[index++] = std::sinh(complex<double>(0., 0.));
        buf_out1_access[index++] = std::cosh(complex<double>(0., 0.));
        buf_out1_access[index++] = std::cosh(complex<double>(INFINITY, 0.));
        buf_out1_access[index++] = std::tanh(complex<double>(0., 0.));
        buf_out1_access[index++] = std::tanh(complex<double>(INFINITY, 1.));
        buf_out1_access[index++] = std::asinh(complex<double>(0., 0.));
        buf_out1_access[index++] = std::asinh(complex<double>(1., INFINITY));
        buf_out1_access[index++] = std::asinh(complex<double>(INFINITY, 1.));
        buf_out1_access[index++] = std::acosh(complex<double>(0., 0.));
        buf_out1_access[index++] = std::acosh(complex<double>(1., INFINITY));
        buf_out1_access[index++] = std::acosh(complex<double>(INFINITY, 1.));
        buf_out1_access[index++] = std::atanh(complex<double>(0., 0.));
        buf_out1_access[index++] = std::atanh(complex<double>(1., INFINITY));
        buf_out1_access[index++] = std::conj(complex<double>(1., 4.));
        buf_out1_access[index++] = std::conj(complex<double>(18., 7.));
        buf_out1_access[index++] = std::tan(complex<double>(1., 0.));
        buf_out1_access[index++] = std::tan(complex<double>(0., 1.));
        buf_out1_access[index++] = std::asin(complex<double>(1., 0.));
        buf_out1_access[index++] = std::atan(complex<double>(0., 2.));

        index = 0;
        buf_out2_access[index++] = std::norm(complex<double>(0., 0.));
        buf_out2_access[index++] = std::norm(complex<double>(3., 4.));
        buf_out2_access[index++] = std::norm(complex<double>(12., 5.));
        buf_out2_access[index++] = std::norm(complex<double>(INFINITY, 1.));
        buf_out2_access[index++] = std::abs(complex<double>(0., 0.));
        buf_out2_access[index++] = std::abs(complex<double>(3., 4.));
        buf_out2_access[index++] = std::abs(complex<double>(12., 5.));
        buf_out2_access[index++] = std::abs(complex<double>(INFINITY, 1.));
        buf_out2_access[index++] = std::arg(complex<double>(1., 0.));
        buf_out2_access[index++] = std::arg(complex<double>(0., 1.));
      });
    });
  }

  for (size_t idx = 0; idx < TestArraySize1; ++idx) {
    assert(approx_equal_cmplx(result1[idx], ref1_results[idx]));
  }
  for (size_t idx = 0; idx < TestArraySize2; ++idx) {
    assert(approx_equal_fp(result2[idx], ref2_results[idx]));
  }
}

int main() {
  s::queue deviceQueue;
  if (deviceQueue.get_device().has(sycl::aspect::fp64)) {
    device_complex_test(deviceQueue);
    std::cout << "Pass" << std::endl;
  }
}
