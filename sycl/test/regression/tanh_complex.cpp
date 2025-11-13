// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

// Checks the results of tanh on certain complex numbers.

#define SYCL_EXT_ONEAPI_COMPLEX

#include <sycl/ext/oneapi/experimental/complex/complex.hpp>
#include <sycl/sycl.hpp>

#include <complex>
#include <limits>

namespace syclexp = sycl::ext::oneapi::experimental;

int Failures = 0;

template <typename T> bool FloatingPointEq(T LHS, T RHS) {
  if (std::isnan(LHS))
    return std::isnan(RHS);
  return LHS == RHS;
}

#define CHECK_TANH_RESULT(REAL, IMAG, T)                                       \
  {                                                                            \
    syclexp::complex<T> sycl_res =                                             \
        syclexp::tanh(syclexp::complex<T>{REAL, IMAG});                        \
    std::complex<T> std_res = std::tanh(std::complex<T>{REAL, IMAG});          \
    if (!FloatingPointEq(sycl_res.real(), std_res.real())) {                   \
      std::cout << "Real differ in tanh((" << REAL << ", " << IMAG             \
                << ")): " << sycl_res.real() << " != " << std_res.real()       \
                << std::endl;                                                  \
      ++Failures;                                                              \
    }                                                                          \
    if (!FloatingPointEq(sycl_res.imag(), std_res.imag())) {                   \
      std::cout << "Imag differ in tanh((" << REAL << ", " << IMAG             \
                << ")): " << sycl_res.imag() << " != " << std_res.imag()       \
                << std::endl;                                                  \
      ++Failures;                                                              \
    }                                                                          \
  }

int main() {
  CHECK_TANH_RESULT(0, -11.0, float);
  CHECK_TANH_RESULT(0, -11.0, double);

  CHECK_TANH_RESULT(std::numeric_limits<float>::infinity(), 32.0, float);
  CHECK_TANH_RESULT(std::numeric_limits<double>::infinity(), 32.0, double);

  CHECK_TANH_RESULT(32.0, std::numeric_limits<float>::infinity(), float);
  CHECK_TANH_RESULT(32.0, std::numeric_limits<double>::infinity(), double);

  CHECK_TANH_RESULT(std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::infinity(), float);
  CHECK_TANH_RESULT(std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity(), double);

  CHECK_TANH_RESULT(-std::numeric_limits<float>::infinity(), 32.0, float);
  CHECK_TANH_RESULT(-std::numeric_limits<double>::infinity(), 32.0, double);

  CHECK_TANH_RESULT(32.0, -std::numeric_limits<float>::infinity(), float);
  CHECK_TANH_RESULT(32.0, -std::numeric_limits<double>::infinity(), double);

  CHECK_TANH_RESULT(-std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity(), float);
  CHECK_TANH_RESULT(-std::numeric_limits<double>::infinity(),
                    -std::numeric_limits<double>::infinity(), double);

  CHECK_TANH_RESULT(std::numeric_limits<float>::max(), 0.0, float);
  CHECK_TANH_RESULT(std::numeric_limits<double>::max(), 0.0, double);

  CHECK_TANH_RESULT(0.0, std::numeric_limits<float>::max(), float);
  CHECK_TANH_RESULT(0.0, std::numeric_limits<double>::max(), double);

  CHECK_TANH_RESULT(0.0, 0.0, float);
  CHECK_TANH_RESULT(0.0, 0.0, double);

  CHECK_TANH_RESULT(std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN(), float);
  CHECK_TANH_RESULT(std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::quiet_NaN(), double);

  CHECK_TANH_RESULT(-std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN(), float);
  CHECK_TANH_RESULT(-std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::quiet_NaN(), double);

  CHECK_TANH_RESULT(std::numeric_limits<float>::quiet_NaN(), 0.0, float);
  CHECK_TANH_RESULT(std::numeric_limits<double>::quiet_NaN(), 0.0, double);

  CHECK_TANH_RESULT(std::numeric_limits<float>::quiet_NaN(), 1.0, float);
  CHECK_TANH_RESULT(std::numeric_limits<double>::quiet_NaN(), 1.0, double);

  CHECK_TANH_RESULT(std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::quiet_NaN(), float);
  CHECK_TANH_RESULT(std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(), double);

  return Failures;
}
