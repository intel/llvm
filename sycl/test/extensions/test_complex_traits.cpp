// RUN: %clangxx -fsycl -fsyntax-only %s

#define SYCL_EXT_ONEAPI_COMPLEX

#include <sycl/ext/oneapi/experimental/sycl_complex.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;

// Check is_gencomplex
void test_is_gencomplex() {
  static_assert(is_gencomplex_v<complex<double>> == true);
  static_assert(is_gencomplex_v<complex<float>> == true);
  static_assert(is_gencomplex_v<complex<sycl::half>> == true);

  static_assert(is_gencomplex_v<complex<long long>> == false);
  static_assert(is_gencomplex_v<complex<long>> == false);
  static_assert(is_gencomplex_v<complex<int>> == false);
  static_assert(is_gencomplex_v<complex<unsigned long long>> == false);
  static_assert(is_gencomplex_v<complex<unsigned long>> == false);
  static_assert(is_gencomplex_v<complex<unsigned int>> == false);
}

// Check is_genfloat
void test_is_genfloat() {
  static_assert(is_genfloat_v<double> == true);
  static_assert(is_genfloat_v<float> == true);
  static_assert(is_genfloat_v<sycl::half> == true);

  static_assert(is_genfloat_v<long long> == false);
  static_assert(is_genfloat_v<long> == false);
  static_assert(is_genfloat_v<int> == false);
  static_assert(is_genfloat_v<unsigned long long> == false);
  static_assert(is_genfloat_v<unsigned long> == false);
  static_assert(is_genfloat_v<unsigned int> == false);
}

// Check is_mgencomplex
void test_is_mgencomplex() {
  static_assert(is_mgencomplex_v<sycl::marray<complex<double>, 42>> == true);
  static_assert(is_mgencomplex_v<sycl::marray<complex<float>, 42>> == true);
  static_assert(is_mgencomplex_v<sycl::marray<complex<sycl::half>, 42>> == true);

  static_assert(is_mgencomplex_v<sycl::marray<complex<long long>, 42>> == false);
  static_assert(is_mgencomplex_v<sycl::marray<complex<long>, 42>> == false);
  static_assert(is_mgencomplex_v<sycl::marray<complex<int>, 42>> == false);
  static_assert(is_mgencomplex_v<sycl::marray<complex<unsigned long long>, 42>> == false);
  static_assert(is_mgencomplex_v<sycl::marray<complex<unsigned long>, 42>> == false);
  static_assert(is_mgencomplex_v<sycl::marray<complex<unsigned int>, 42>> == false);
}

// Check is_plus
void test_is_plus() {
  static_assert(cplx::detail::is_plus_v<sycl::plus<>> == true);

  static_assert(cplx::detail::is_plus_v<sycl::multiplies<>> == false);
  static_assert(cplx::detail::is_plus_v<sycl::bit_and<>> == false);
  static_assert(cplx::detail::is_plus_v<sycl::logical_and<>> == false);
  static_assert(cplx::detail::is_plus_v<sycl::minimum<>> == false);
  static_assert(cplx::detail::is_plus_v<sycl::maximum<>> == false);
}

// Check is_multiplies
void test_is_multiplies() {
  static_assert(cplx::detail::is_multiplies_v<sycl::multiplies<>> == true);

  static_assert(cplx::detail::is_multiplies_v<sycl::plus<>> == false);
  static_assert(cplx::detail::is_multiplies_v<sycl::bit_and<>> == false);
  static_assert(cplx::detail::is_multiplies_v<sycl::logical_and<>> == false);
  static_assert(cplx::detail::is_multiplies_v<sycl::minimum<>> == false);
  static_assert(cplx::detail::is_multiplies_v<sycl::maximum<>> == false);
}

// Check is_binary_op_supported
void test_is_binary_op_supported() {
  static_assert(cplx::detail::is_binary_op_supported_v<sycl::plus<>> == true);
  static_assert(cplx::detail::is_binary_op_supported_v<sycl::multiplies<>> == true);

  static_assert(cplx::detail::is_binary_op_supported_v<sycl::bit_and<>> == false);
  static_assert(cplx::detail::is_binary_op_supported_v<sycl::logical_and<>> == false);
  static_assert(cplx::detail::is_binary_op_supported_v<sycl::minimum<>> == false);
  static_assert(cplx::detail::is_binary_op_supported_v<sycl::maximum<>> == false);
}

int main() {
  test_is_gencomplex();
  test_is_genfloat();
  test_is_mgencomplex();
  test_is_plus();
  test_is_multiplies();
  test_is_binary_op_supported();

  return 0;
}
