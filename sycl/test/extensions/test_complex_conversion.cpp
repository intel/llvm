// RUN: %clangxx -fsycl -fsyntax-only %s

#define SYCL_EXT_ONEAPI_COMPLEX

#include <array>
#include <cmath>

#include <sycl/ext/oneapi/experimental/sycl_complex.hpp>
#include <sycl/sycl.hpp>

// Helper for passing infinity and nan values
template <typename T>
inline constexpr T inf_val = std::numeric_limits<T>::infinity();
template <typename T>
inline constexpr T nan_val = std::numeric_limits<T>::quiet_NaN();

// Check conversion sycl:complex to std::complex
template <typename T>
void test_sycl_complex_to_std_complex() {
  auto arr = std::array<sycl::ext::oneapi::experimental::complex<T>, 8>{
      sycl::ext::oneapi::experimental::complex<T>{0, 0},
      sycl::ext::oneapi::experimental::complex<T>{-0, 0},
      sycl::ext::oneapi::experimental::complex<T>{0, -0},
      sycl::ext::oneapi::experimental::complex<T>{-0, -0},
      sycl::ext::oneapi::experimental::complex<T>{1, 1},
      sycl::ext::oneapi::experimental::complex<T>{-1, 1},
      sycl::ext::oneapi::experimental::complex<T>{1, -1},
      sycl::ext::oneapi::experimental::complex<T>{-1, -1},
  };

  for (const auto &lhs : arr) {
    const auto rhs = static_cast<std::complex<T>>(lhs);

    assert(lhs.real() == rhs.real() && "sycl::complex differs from std::complex after conversion");
    assert(lhs.imag() == rhs.imag() && "sycl::complex differs from std::complex after conversion");
  }
}

// Check edge-cases conversion sycl:complex to std::complex
template <typename T>
void test_edge_case_sycl_complex_to_std_complex() {
  {
    auto lhs = sycl::ext::oneapi::experimental::complex<T>{inf_val<T>, inf_val<T>};
    auto rhs = static_cast<std::complex<T>>(lhs);

    assert(std::isinf(lhs.real()) && std::isinf(rhs.real()) &&
           "sycl::complex differs from std::complex after conversion");
    assert(std::isinf(lhs.imag()) && std::isinf(rhs.imag()) &&
           "sycl::complex differs from std::complex after conversion");
  }
  {
    auto lhs = sycl::ext::oneapi::experimental::complex<T>{inf_val<T>, nan_val<T>};
    auto rhs = static_cast<std::complex<T>>(lhs);

    assert(std::isinf(lhs.real()) && std::isinf(rhs.real()) &&
           "sycl::complex differs from std::complex after conversion");
    assert(std::isnan(lhs.imag()) && std::isnan(rhs.imag()) &&
           "sycl::complex differs from std::complex after conversion");
  }
  {
    auto lhs = sycl::ext::oneapi::experimental::complex<T>{nan_val<T>, inf_val<T>};
    auto rhs = static_cast<std::complex<T>>(lhs);

    assert(std::isnan(lhs.real()) && std::isnan(rhs.real()) &&
           "sycl::complex differs from std::complex after conversion");
    assert(std::isinf(lhs.imag()) && std::isinf(rhs.imag()) &&
           "sycl::complex differs from std::complex after conversion");
  }
  {
    auto lhs = sycl::ext::oneapi::experimental::complex<T>{nan_val<T>, nan_val<T>};
    auto rhs = static_cast<std::complex<T>>(lhs);

    assert(std::isnan(lhs.real()) && std::isnan(rhs.real()) &&
           "sycl::complex differs from std::complex after conversion");
    assert(std::isnan(lhs.imag()) && std::isnan(rhs.imag()) &&
           "sycl::complex differs from std::complex after conversion");
  }
}

int main() {
    test_sycl_complex_to_std_complex<double>();
    test_sycl_complex_to_std_complex<float>();
    test_sycl_complex_to_std_complex<sycl::half>();

    test_edge_case_sycl_complex_to_std_complex<double>();
    test_edge_case_sycl_complex_to_std_complex<float>();
    test_edge_case_sycl_complex_to_std_complex<sycl::half>();

    return 0;
}
