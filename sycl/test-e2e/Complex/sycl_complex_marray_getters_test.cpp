// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "sycl_complex_marray_test_cases.hpp"

template <typename T> struct test_real {
  bool operator()(sycl::queue &Q, const std::vector<double> &init,
                  const std::vector<double> &ref = {}, bool use_ref = false) {
    bool pass = true;

    /* std::complex test cases */
    sycl::marray<T, GETTERS_TEST_CASE_SIZE> std_in;
    /* sycl::complex test cases */
    sycl::marray<experimental::complex<T>, GETTERS_TEST_CASE_SIZE> cplx_in;

    for (std::size_t i = 0; i < GETTERS_TEST_CASE_SIZE; ++i) {
      std_in[i] = static_cast<T>(init[i]);
      cplx_in[i] =
          experimental::complex<T>{static_cast<T>(init[i]), static_cast<T>(0)};
    }

    sycl::buffer<sycl::marray<T, GETTERS_TEST_CASE_SIZE>> cplx_out_buf{
        sycl::range{1}};

    Q.submit([&](sycl::handler &cgh) {
      sycl::accessor cplx_out{cplx_out_buf, cgh};

      cgh.single_task([=]() {
        cplx_out[0] = sycl::ext::oneapi::experimental::real(cplx_in);
      });
    });

    /* Check cplx::complex output from device */
    sycl::host_accessor cplx_out_acc{cplx_out_buf};
    pass &= check_results(cplx_out_acc[0], std_in, /*is_device*/ true);

    /* Check cplx::complex output from host */
    cplx_out_acc[0] = sycl::ext::oneapi::experimental::real(cplx_in);
    pass &= check_results(cplx_out_acc[0], std_in, /*is_device*/ false);

    return pass;
  }
};

template <typename T> struct test_imag {
  bool operator()(sycl::queue &Q, const std::vector<double> &init,
                  const std::vector<double> &ref = {}, bool use_ref = false) {
    bool pass = true;

    /* std::complex test cases */
    sycl::marray<T, GETTERS_TEST_CASE_SIZE> std_in;
    /* sycl::complex test cases */
    sycl::marray<experimental::complex<T>, GETTERS_TEST_CASE_SIZE> cplx_in;

    for (std::size_t i = 0; i < GETTERS_TEST_CASE_SIZE; ++i) {
      std_in[i] = static_cast<T>(init[i]);
      cplx_in[i] =
          experimental::complex<T>{static_cast<T>(0), static_cast<T>(init[i])};
    }

    sycl::buffer<sycl::marray<T, GETTERS_TEST_CASE_SIZE>> cplx_out_buf{
        sycl::range{1}};

    Q.submit([&](sycl::handler &cgh) {
      sycl::accessor cplx_out{cplx_out_buf, cgh};

      cgh.single_task([=]() {
        cplx_out[0] = sycl::ext::oneapi::experimental::imag(cplx_in);
      });
    });

    /* Check cplx::complex output from device */
    sycl::host_accessor cplx_out_acc{cplx_out_buf};
    pass &= check_results(cplx_out_acc[0], std_in, /*is_device*/ true);

    /* Check cplx::complex output from host */
    cplx_out_acc[0] = sycl::ext::oneapi::experimental::imag(cplx_in);
    pass &= check_results(cplx_out_acc[0], std_in, /*is_device*/ false);

    return pass;
  }
};

int main() {
  sycl::queue Q;

  bool test_passes = true;

  /* Test real getter */

  {
    marray_scalar_test_cases<test_real> test;
    test_passes &= test(Q);
  }

  /* Test imag getter */

  {
    marray_scalar_test_cases<test_imag> test;
    test_passes &= test(Q);
  }

  return !test_passes;
}
