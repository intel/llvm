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
  // Allow some rounding differences, but minimal.
  return std::abs(LHS - RHS) < T{0.0001};
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

#define CHECK_TANH_REF_RESULT(REAL, IMAG, REF_REAL, REF_IMAG, T)               \
  {                                                                            \
    syclexp::complex<T> sycl_res =                                             \
        syclexp::tanh(syclexp::complex<T>{REAL, IMAG});                        \
    if (!FloatingPointEq(sycl_res.real(), T{REF_REAL})) {                      \
      std::cout << "Real differ in tanh((" << REAL << ", " << IMAG             \
                << ")): " << sycl_res.real() << " != " << REF_REAL             \
                << std::endl;                                                  \
      ++Failures;                                                              \
    }                                                                          \
    if (!FloatingPointEq(sycl_res.imag(), T{REF_IMAG})) {                      \
      std::cout << "Imag differ in tanh((" << REAL << ", " << IMAG             \
                << ")): " << sycl_res.imag() << " != " << REF_IMAG             \
                << std::endl;                                                  \
      ++Failures;                                                              \
    }                                                                          \
  }

int main() {
  // Set precision for easier debugging.
  std::cout << std::setprecision(10);

  CHECK_TANH_RESULT(0, -11.0, float);
  CHECK_TANH_RESULT(0, -11.0, double);

  CHECK_TANH_RESULT(32.0, std::numeric_limits<float>::infinity(), float);
  CHECK_TANH_RESULT(32.0, std::numeric_limits<double>::infinity(), double);

  CHECK_TANH_RESULT(32.0, -std::numeric_limits<float>::infinity(), float);
  CHECK_TANH_RESULT(32.0, -std::numeric_limits<double>::infinity(), double);

  CHECK_TANH_RESULT(std::numeric_limits<float>::max(), 0.0, float);
  CHECK_TANH_RESULT(std::numeric_limits<double>::max(), 0.0, double);

  CHECK_TANH_RESULT(0.0, std::numeric_limits<float>::max(), float);
  CHECK_TANH_RESULT(0.0, std::numeric_limits<double>::max(), double);

  CHECK_TANH_RESULT(0.0, 0.0, float);
  CHECK_TANH_RESULT(0.0, 0.0, double);

  CHECK_TANH_RESULT(std::numeric_limits<float>::quiet_NaN(), 1.0, float);
  CHECK_TANH_RESULT(std::numeric_limits<double>::quiet_NaN(), 1.0, double);

  CHECK_TANH_RESULT(std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::quiet_NaN(), float);
  CHECK_TANH_RESULT(std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(), double);

  // The MSVC implementation of tanh for complex numbers does not adhere to the
  // following requirements set by the definition of std::tanh:
  //  * When the input has an infinite real, then the function should return
  //    (1, +-0).
  //  * When the input is (NaN, 0), the result should be (NaN, 0).
  // Instead we check the results using reference values rather than trusting
  // the result of std::tanh in these cases.
  CHECK_TANH_REF_RESULT(std::numeric_limits<float>::infinity(), 32.0, 1.0, 0.0,
                        float);
  CHECK_TANH_REF_RESULT(std::numeric_limits<double>::infinity(), 32.0, 1.0, 0.0,
                        double);

  CHECK_TANH_REF_RESULT(std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::infinity(), 1, 0.0, float);
  CHECK_TANH_REF_RESULT(std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity(), 1, 0.0,
                        double);

  CHECK_TANH_REF_RESULT(-std::numeric_limits<float>::infinity(), 32.0, -1.0,
                        0.0, float);
  CHECK_TANH_REF_RESULT(-std::numeric_limits<double>::infinity(), 32.0, -1.0,
                        0.0, double);

  CHECK_TANH_REF_RESULT(-std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity(), -1.0, 0.0,
                        float);
  CHECK_TANH_REF_RESULT(-std::numeric_limits<double>::infinity(),
                        -std::numeric_limits<double>::infinity(), -1.0, 0.0,
                        double);

  CHECK_TANH_REF_RESULT(std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN(), 1.0, 0.0,
                        float);
  CHECK_TANH_REF_RESULT(std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::quiet_NaN(), 1.0, 0.0,
                        double);

  CHECK_TANH_REF_RESULT(-std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN(), -1.0, 0.0,
                        float);
  CHECK_TANH_REF_RESULT(-std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::quiet_NaN(), -1.0, 0.0,
                        double);

  CHECK_TANH_REF_RESULT(std::numeric_limits<float>::quiet_NaN(), 0.0,
                        std::numeric_limits<float>::quiet_NaN(), 0.0, float);
  CHECK_TANH_REF_RESULT(std::numeric_limits<double>::quiet_NaN(), 0.0,
                        std::numeric_limits<float>::quiet_NaN(), 0.0, double);

  return Failures;
}
