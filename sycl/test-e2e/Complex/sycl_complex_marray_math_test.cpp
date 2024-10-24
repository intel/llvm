// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "sycl_complex_marray_test_cases.hpp"

#define TEST_MATH_MARRAY_CPLX_IN_CPLX_OUT(func_name)                           \
  template <typename T> struct test_##func_name {                              \
    bool operator()(sycl::queue &Q, const std::vector<cmplx<double>> &init,    \
                    const std::vector<cmplx<double>> &ref = {},                \
                    bool use_ref = false) {                                    \
      bool pass = true;                                                        \
      using X = typename std::conditional<std::is_same<T, sycl::half>::value,  \
                                          float, T>::type;                     \
                                                                               \
      /* std::complex test cases */                                            \
      sycl::marray<std::complex<X>, DEFAULT_TEST_CASE_SIZE> std_in;            \
      /* sycl::complex test cases */                                           \
      sycl::marray<experimental::complex<T>, DEFAULT_TEST_CASE_SIZE> cplx_in;  \
                                                                               \
      for (std::size_t i = 0; i < DEFAULT_TEST_CASE_SIZE; ++i) {               \
        std_in[i] = init_std_complex<T>(init[i].re, init[i].im);               \
        cplx_in[i] = experimental::complex<T>{static_cast<T>(init[i].re),      \
                                              static_cast<T>(init[i].im)};     \
      }                                                                        \
                                                                               \
      sycl::marray<std::complex<X>, DEFAULT_TEST_CASE_SIZE> std_out{};         \
      sycl::buffer<                                                            \
          sycl::marray<experimental::complex<T>, DEFAULT_TEST_CASE_SIZE>>      \
          cplx_out_buf{sycl::range{1}};                                        \
                                                                               \
      /* Get std::complex output */                                            \
      if (use_ref) {                                                           \
        for (std::size_t i = 0; i < DEFAULT_TEST_CASE_SIZE; ++i) {             \
          T re = static_cast<T>(ref[i].re);                                    \
          T im = static_cast<T>(ref[i].im);                                    \
          std_out[i] = std::complex<T>{re, im};                                \
        }                                                                      \
      } else {                                                                 \
        for (std::size_t i = 0; i < DEFAULT_TEST_CASE_SIZE; ++i) {             \
          std_out[i] = std::func_name(std_in[i]);                              \
        }                                                                      \
      }                                                                        \
                                                                               \
      Q.submit([&](sycl::handler &cgh) {                                       \
        sycl::accessor cplx_out{cplx_out_buf, cgh};                            \
                                                                               \
        cgh.single_task(                                                       \
            [=]() { cplx_out[0] = experimental::func_name(cplx_in); });        \
      });                                                                      \
                                                                               \
      /* Check cplx::complex output from device */                             \
      sycl::host_accessor cplx_out_acc{cplx_out_buf};                          \
      pass &= check_results(cplx_out_acc[0], convert_marray<T>(std_out),       \
                            /*is_device*/ true);                               \
                                                                               \
      /* Check cplx::complex output from host */                               \
      cplx_out_acc[0] = experimental::func_name(cplx_in);                      \
      pass &= check_results(cplx_out_acc[0], convert_marray<T>(std_out),       \
                            /*is_device*/ false);                              \
                                                                               \
      return pass;                                                             \
    }                                                                          \
  };

TEST_MATH_MARRAY_CPLX_IN_CPLX_OUT(acos)
TEST_MATH_MARRAY_CPLX_IN_CPLX_OUT(asin)
TEST_MATH_MARRAY_CPLX_IN_CPLX_OUT(atan)
TEST_MATH_MARRAY_CPLX_IN_CPLX_OUT(acosh)
TEST_MATH_MARRAY_CPLX_IN_CPLX_OUT(asinh)
TEST_MATH_MARRAY_CPLX_IN_CPLX_OUT(atanh)
TEST_MATH_MARRAY_CPLX_IN_CPLX_OUT(conj)
TEST_MATH_MARRAY_CPLX_IN_CPLX_OUT(cos)
TEST_MATH_MARRAY_CPLX_IN_CPLX_OUT(cosh)
TEST_MATH_MARRAY_CPLX_IN_CPLX_OUT(exp)
TEST_MATH_MARRAY_CPLX_IN_CPLX_OUT(log)
TEST_MATH_MARRAY_CPLX_IN_CPLX_OUT(log10)
TEST_MATH_MARRAY_CPLX_IN_CPLX_OUT(proj)
TEST_MATH_MARRAY_CPLX_IN_CPLX_OUT(sin)
TEST_MATH_MARRAY_CPLX_IN_CPLX_OUT(sinh)
TEST_MATH_MARRAY_CPLX_IN_CPLX_OUT(sqrt)
TEST_MATH_MARRAY_CPLX_IN_CPLX_OUT(tan)
TEST_MATH_MARRAY_CPLX_IN_CPLX_OUT(tanh)

#undef TEST_MATH_MARRAY_CPLX_IN_CPLX_OUT

#define TEST_MATH_MARRAY_CPLX_IN_DECI_OUT(func_name)                           \
  template <typename T> struct test_##func_name {                              \
    bool operator()(sycl::queue &Q, const std::vector<cmplx<double>> &init,    \
                    const std::vector<cmplx<double>> &ref = {},                \
                    bool use_ref = false) {                                    \
      bool pass = true;                                                        \
                                                                               \
      using X = typename std::conditional<std::is_same<T, sycl::half>::value,  \
                                          float, T>::type;                     \
                                                                               \
      /* std::complex test cases */                                            \
      sycl::marray<std::complex<X>, DEFAULT_TEST_CASE_SIZE> std_in;            \
      /* sycl::complex test cases */                                           \
      sycl::marray<experimental::complex<T>, DEFAULT_TEST_CASE_SIZE> cplx_in;  \
                                                                               \
      for (std::size_t i = 0; i < DEFAULT_TEST_CASE_SIZE; ++i) {               \
        std_in[i] = init_std_complex<T>(init[i].re, init[i].im);               \
        cplx_in[i] = experimental::complex<T>{static_cast<T>(init[i].re),      \
                                              static_cast<T>(init[i].im)};     \
      }                                                                        \
                                                                               \
      sycl::marray<T, DEFAULT_TEST_CASE_SIZE> std_out{};                       \
      sycl::buffer<sycl::marray<T, DEFAULT_TEST_CASE_SIZE>> cplx_out_buf{      \
          sycl::range{1}};                                                     \
                                                                               \
      /* Get std::complex output */                                            \
      if (use_ref) {                                                           \
        for (std::size_t i = 0; i < DEFAULT_TEST_CASE_SIZE; ++i) {             \
          std_out[i] = static_cast<T>(ref[i].re);                              \
        }                                                                      \
      } else {                                                                 \
        for (std::size_t i = 0; i < DEFAULT_TEST_CASE_SIZE; ++i) {             \
          std_out[i] = std::func_name(std_in[i]);                              \
        }                                                                      \
      }                                                                        \
                                                                               \
      Q.submit([&](sycl::handler &cgh) {                                       \
        sycl::accessor cplx_out{cplx_out_buf, cgh};                            \
                                                                               \
        cgh.single_task(                                                       \
            [=]() { cplx_out[0] = experimental::func_name(cplx_in); });        \
      });                                                                      \
                                                                               \
      /* Check cplx::complex output from device */                             \
      sycl::host_accessor cplx_out_acc{cplx_out_buf};                          \
      pass &= check_results(cplx_out_acc[0], std_out, /*is_device*/ true);     \
                                                                               \
      /* Check cplx::complex output from host */                               \
      cplx_out_acc[0] = experimental::func_name(cplx_in);                      \
      pass &= check_results(cplx_out_acc[0], std_out, /*is_device*/ false);    \
                                                                               \
      return pass;                                                             \
    }                                                                          \
  };

TEST_MATH_MARRAY_CPLX_IN_DECI_OUT(abs)
TEST_MATH_MARRAY_CPLX_IN_DECI_OUT(arg)
TEST_MATH_MARRAY_CPLX_IN_DECI_OUT(norm)

#undef TEST_MARRAY_CPLX_IN_DECI_OUT

template <typename T> struct test_polar {
  bool operator()(sycl::queue &Q, const std::vector<cmplx<double>> &init,
                  const std::vector<cmplx<double>> &ref = {},
                  bool use_ref = false) {
    bool pass = true;

    /* test cases */
    sycl::marray<T, POLAR_TEST_CASE_SIZE> rho;
    sycl::marray<T, POLAR_TEST_CASE_SIZE> theta;
    for (std::size_t i = 0; i < POLAR_TEST_CASE_SIZE; ++i) {
      rho[i] = static_cast<T>(init[i].re);
      theta[i] = static_cast<T>(init[i].im);
    }

    sycl::marray<std::complex<T>, POLAR_TEST_CASE_SIZE> std_out{};
    sycl::buffer<sycl::marray<experimental::complex<T>, POLAR_TEST_CASE_SIZE>>
        cplx_out_buf{sycl::range{1}};

    /* Get std::complex output */
    if (use_ref) {
      for (std::size_t i = 0; i < POLAR_TEST_CASE_SIZE; ++i) {
        T re = static_cast<T>(ref[i].re);
        T im = static_cast<T>(ref[i].im);
        std_out[i] = std::complex<T>{re, im};
      }
    } else {
      for (std::size_t i = 0; i < POLAR_TEST_CASE_SIZE; ++i) {
        std_out[i] = std::polar(rho[i], theta[i]);
      }
    }

    Q.submit([&](sycl::handler &cgh) {
      sycl::accessor cplx_out{cplx_out_buf, cgh};

      cgh.single_task([=]() { cplx_out[0] = experimental::polar(rho, theta); });
    });

    /* Check cplx::complex output from device */
    sycl::host_accessor cplx_out_acc{cplx_out_buf};
    pass &= check_results(cplx_out_acc[0], std_out, /*is_device*/ true);

    /* Check cplx::complex output from host */
    cplx_out_acc[0] = experimental::polar(rho, theta);
    pass &= check_results(cplx_out_acc[0], std_out, /*is_device*/ false);

    return pass;
  }
};

template <typename T> struct test_pow_cplx_cplx {
  bool operator()(sycl::queue &Q, const std::vector<cmplx<double>> &init,
                  const std::vector<cmplx<double>> &ref = {},
                  bool use_ref = false) {
    bool pass = true;

    using X = typename std::conditional<std::is_same<T, sycl::half>::value,
                                        float, T>::type;

    /* std::complex test cases */
    sycl::marray<std::complex<X>, DEFAULT_TEST_CASE_SIZE> std_in;
    /* sycl::complex test cases */
    sycl::marray<experimental::complex<T>, DEFAULT_TEST_CASE_SIZE> cplx_in;

    for (std::size_t i = 0; i < DEFAULT_TEST_CASE_SIZE; ++i) {
      std_in[i] = init_std_complex<T>(init[i].re, init[i].im);
      cplx_in[i] = experimental::complex<T>{static_cast<T>(init[i].re),
                                            static_cast<T>(init[i].im)};
    }

    sycl::marray<std::complex<X>, DEFAULT_TEST_CASE_SIZE> std_out{};
    sycl::buffer<sycl::marray<experimental::complex<T>, DEFAULT_TEST_CASE_SIZE>>
        cplx_out_buf{sycl::range{1}};

    /* Get std::complex output */
    if (use_ref) {
      for (std::size_t i = 0; i < DEFAULT_TEST_CASE_SIZE; ++i) {
        T re = static_cast<T>(ref[i].re);
        T im = static_cast<T>(ref[i].im);
        std_out[i] = std::complex<T>{re, im};
      }
    } else {
      for (std::size_t i = 0; i < DEFAULT_TEST_CASE_SIZE; ++i) {
        std_out[i] = std::pow(std_in[i], std_in[i]);
      }
    }

    Q.submit([&](sycl::handler &cgh) {
      sycl::accessor cplx_out{cplx_out_buf, cgh};

      cgh.single_task(
          [=]() { cplx_out[0] = experimental::pow(cplx_in, cplx_in); });
    });

    /* Check cplx::complex output from device */
    sycl::host_accessor cplx_out_acc{cplx_out_buf};
    pass &= check_results(cplx_out_acc[0], convert_marray<T>(std_out),
                          /*is_device*/ true);

    /* Check cplx::complex output from host */
    cplx_out_acc[0] = experimental::pow(cplx_in, cplx_in);
    pass &= check_results(cplx_out_acc[0], convert_marray<T>(std_out),
                          /*is_device*/ false);

    return pass;
  }
};

template <typename T> struct test_pow_cplx_deci {
  bool operator()(sycl::queue &Q, const std::vector<cmplx<double>> &init,
                  const std::vector<cmplx<double>> &ref = {},
                  bool use_ref = false) {
    bool pass = true;

    using X = typename std::conditional<std::is_same<T, sycl::half>::value,
                                        float, T>::type;

    /* std::complex test cases */
    sycl::marray<std::complex<X>, DEFAULT_TEST_CASE_SIZE> std_in;
    /* sycl::complex test cases */
    sycl::marray<experimental::complex<T>, DEFAULT_TEST_CASE_SIZE> cplx_in;

    for (std::size_t i = 0; i < DEFAULT_TEST_CASE_SIZE; ++i) {
      std_in[i] = init_std_complex<T>(init[i].re, init[i].im);
      cplx_in[i] = experimental::complex<T>{static_cast<T>(init[i].re),
                                            static_cast<T>(init[i].im)};
    }

    sycl::marray<std::complex<X>, DEFAULT_TEST_CASE_SIZE> std_out{};
    sycl::buffer<sycl::marray<experimental::complex<T>, DEFAULT_TEST_CASE_SIZE>>
        cplx_out_buf{sycl::range{1}};

    /* Get std::complex output */
    if (use_ref) {
      for (std::size_t i = 0; i < DEFAULT_TEST_CASE_SIZE; ++i) {
        T re = static_cast<T>(ref[i].re);
        T im = static_cast<T>(ref[i].im);
        std_out[i] = std::complex<T>{re, im};
      }
    } else {
      for (std::size_t i = 0; i < DEFAULT_TEST_CASE_SIZE; ++i) {
        std_out[i] = std::pow(std_in[i], std_in[i].real());
      }
    }

    Q.submit([&](sycl::handler &cgh) {
      sycl::accessor cplx_out{cplx_out_buf, cgh};

      cgh.single_task([=]() {
        cplx_out[0] = experimental::pow(
            cplx_in, sycl::ext::oneapi::experimental::real(cplx_in));
      });
    });

    /* Check cplx::complex output from device */
    sycl::host_accessor cplx_out_acc{cplx_out_buf};
    pass &= check_results(cplx_out_acc[0], convert_marray<T>(std_out),
                          /*is_device*/ true);

    /* Check cplx::complex output from host */
    cplx_out_acc[0] = experimental::pow(
        cplx_in, sycl::ext::oneapi::experimental::real(cplx_in));
    pass &= check_results(cplx_out_acc[0], convert_marray<T>(std_out),
                          /*is_device*/ false);

    return pass;
  }
};

template <typename T> struct test_pow_deci_cplx {
  bool operator()(sycl::queue &Q, const std::vector<cmplx<double>> &init,
                  const std::vector<cmplx<double>> &ref = {},
                  bool use_ref = false) {
    bool pass = true;

    using X = typename std::conditional<std::is_same<T, sycl::half>::value,
                                        float, T>::type;

    /* std::complex test cases */
    sycl::marray<std::complex<X>, DEFAULT_TEST_CASE_SIZE> std_in;
    /* sycl::complex test cases */
    sycl::marray<experimental::complex<T>, DEFAULT_TEST_CASE_SIZE> cplx_in;

    for (std::size_t i = 0; i < DEFAULT_TEST_CASE_SIZE; ++i) {
      std_in[i] = init_std_complex<T>(init[i].re, init[i].im);
      cplx_in[i] = experimental::complex<T>{static_cast<T>(init[i].re),
                                            static_cast<T>(init[i].im)};
    }

    sycl::marray<std::complex<X>, DEFAULT_TEST_CASE_SIZE> std_out{};
    sycl::buffer<sycl::marray<experimental::complex<T>, DEFAULT_TEST_CASE_SIZE>>
        cplx_out_buf{sycl::range{1}};

    /* Get std::complex output */
    if (use_ref) {
      for (std::size_t i = 0; i < DEFAULT_TEST_CASE_SIZE; ++i) {
        T re = static_cast<T>(ref[i].re);
        T im = static_cast<T>(ref[i].im);
        std_out[i] = std::complex<T>{re, im};
      }
    } else {
      for (std::size_t i = 0; i < DEFAULT_TEST_CASE_SIZE; ++i) {
        std_out[i] = std::pow(std_in[i].real(), std_in[i]);
      }
    }

    Q.submit([&](sycl::handler &cgh) {
      sycl::accessor cplx_out{cplx_out_buf, cgh};

      cgh.single_task([=]() {
        cplx_out[0] = experimental::pow(
            sycl::ext::oneapi::experimental::real(cplx_in), cplx_in);
      });
    });

    /* Check cplx::complex output from device */
    sycl::host_accessor cplx_out_acc{cplx_out_buf};
    pass &= check_results(cplx_out_acc[0], convert_marray<T>(std_out),
                          /*is_device*/ true);

    /* Check cplx::complex output from host */
    cplx_out_acc[0] = experimental::pow(
        sycl::ext::oneapi::experimental::real(cplx_in), cplx_in);
    pass &= check_results(cplx_out_acc[0], convert_marray<T>(std_out),
                          /*is_device*/ false);

    return pass;
  }
};

int main() {
  sycl::queue Q;

  bool test_passes = true;

  /* Test complex in, complex out functions */

  {
    marray_cplx_test_cases<test_acos> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_asin> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_atan> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_acosh> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_asinh> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_atanh> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_conj> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_cos> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_cosh> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_exp> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_log> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_log10> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_proj> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_sin> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_sinh> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_sqrt> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_tan> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_tanh> test;
    test_passes &= test(Q);
  }

  /* Test complex in, decimal out functions */

  {
    marray_cplx_test_cases<test_abs> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_arg> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_norm> test;
    test_passes &= test(Q);
  }

  /* Test polar function */

  {
    marray_cplx_test_cases<test_polar> test;
    test_passes &= test(Q);
  }

  /* Test pow function */

  {
    marray_cplx_test_cases<test_pow_cplx_cplx> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_pow_cplx_deci> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_pow_deci_cplx> test;
    test_passes &= test(Q);
  }

  return !test_passes;
}
