// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-device-code-split=per_kernel %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

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

    auto *cplx_out =
        sycl::malloc_shared<sycl::marray<T, GETTERS_TEST_CASE_SIZE>>(1, Q);

    /* Check cplx::complex output from device */
    Q.single_task([=]() { *cplx_out = cplx_in.real(); }).wait();
    pass &= check_results(*cplx_out, std_in, /*is_device*/ true);

    /* Check cplx::complex output from host */
    *cplx_out = cplx_in.real();
    pass &= check_results(*cplx_out, std_in, /*is_device*/ false);

    sycl::free(cplx_out, Q);

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

    auto *cplx_out =
        sycl::malloc_shared<sycl::marray<T, GETTERS_TEST_CASE_SIZE>>(1, Q);

    /* Check cplx::complex output from device */
    Q.single_task([=]() { *cplx_out = cplx_in.imag(); }).wait();
    pass &= check_results(*cplx_out, std_in, /*is_device*/ true);

    /* Check cplx::complex output from host */
    *cplx_out = cplx_in.imag();
    pass &= check_results(*cplx_out, std_in, /*is_device*/ false);

    sycl::free(cplx_out, Q);

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
