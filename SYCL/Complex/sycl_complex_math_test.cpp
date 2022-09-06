// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-device-code-split=per_kernel %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include "sycl_complex_helper.hpp"
#include "sycl_complex_math_test_cases.hpp"

// Macro for testing complex in, complex out functions

#define TEST_MATH_OP_TYPE(math_func)                                           \
  template <typename T> struct test_##math_func {                              \
    bool operator()(sycl::queue &Q, cmplx<T> init,                             \
                    cmplx<T> ref = cmplx<T>(0, 0), bool use_ref = false) {     \
      bool pass = true;                                                        \
                                                                               \
      auto std_in = init_std_complex(init.re, init.im);                        \
      experimental::complex<T> cplx_input{init.re, init.im};                   \
                                                                               \
      auto *cplx_out = sycl::malloc_shared<experimental::complex<T>>(1, Q);    \
                                                                               \
      /*Get std::complex output*/                                              \
      std::complex<T> std_out{ref.re, ref.im};                                 \
      if (!use_ref)                                                            \
        std_out = std::math_func(std_in);                                      \
                                                                               \
      /*Check cplx::complex output from device*/                               \
      Q.single_task([=]() {                                                    \
         cplx_out[0] = experimental::math_func<T>(cplx_input);                 \
       }).wait();                                                              \
                                                                               \
      pass &= check_results(cplx_out[0], std_out, /*is_device*/ true);         \
                                                                               \
      /*Check cplx::complex output from host*/                                 \
      cplx_out[0] = experimental::math_func<T>(cplx_input);                    \
                                                                               \
      pass &= check_results(cplx_out[0], std_out, /*is_device*/ false);        \
                                                                               \
      sycl::free(cplx_out, Q);                                                 \
                                                                               \
      return pass;                                                             \
    }                                                                          \
  };

TEST_MATH_OP_TYPE(acos)
TEST_MATH_OP_TYPE(asin)
TEST_MATH_OP_TYPE(atan)
TEST_MATH_OP_TYPE(acosh)
TEST_MATH_OP_TYPE(asinh)
TEST_MATH_OP_TYPE(atanh)
TEST_MATH_OP_TYPE(conj)
TEST_MATH_OP_TYPE(cos)
TEST_MATH_OP_TYPE(cosh)
TEST_MATH_OP_TYPE(exp)
TEST_MATH_OP_TYPE(log)
TEST_MATH_OP_TYPE(log10)
TEST_MATH_OP_TYPE(proj)
TEST_MATH_OP_TYPE(sin)
TEST_MATH_OP_TYPE(sinh)
TEST_MATH_OP_TYPE(sqrt)
TEST_MATH_OP_TYPE(tan)
TEST_MATH_OP_TYPE(tanh)

#undef TEST_MATH_OP_TYPE

// Macro for testing complex in, decimal out functions

#define TEST_MATH_OP_TYPE(math_func)                                           \
  template <typename T> struct test_##math_func {                              \
    bool operator()(sycl::queue &Q, cmplx<T> init,                             \
                    cmplx<T> ref = cmplx<T>(0, 0), bool use_ref = false) {     \
      bool pass = true;                                                        \
                                                                               \
      auto std_in = init_std_complex(init.re, init.im);                        \
      experimental::complex<T> cplx_input{init.re, init.im};                   \
                                                                               \
      auto *cplx_out = sycl::malloc_shared<T>(1, Q);                           \
                                                                               \
      /*Get std::complex output*/                                              \
      T std_out = ref.re;                                                      \
      if (!use_ref)                                                            \
        std_out = std::math_func(std_in);                                      \
                                                                               \
      /*Check cplx::complex output from device*/                               \
      Q.single_task([=]() {                                                    \
         cplx_out[0] = experimental::math_func<T>(cplx_input);                 \
       }).wait();                                                              \
                                                                               \
      pass &= check_results(cplx_out[0], std_out, /*is_device*/ true);         \
                                                                               \
      /*Check cplx::complex output from host*/                                 \
      cplx_out[0] = experimental::math_func<T>(cplx_input);                    \
                                                                               \
      pass &= check_results(cplx_out[0], std_out, /*is_device*/ false);        \
                                                                               \
      sycl::free(cplx_out, Q);                                                 \
                                                                               \
      return pass;                                                             \
    }                                                                          \
  };

TEST_MATH_OP_TYPE(abs)
TEST_MATH_OP_TYPE(arg)
TEST_MATH_OP_TYPE(norm)

#undef TEST_MATH_OP_TYPE

// Test for polar function
// The real component is treated as radius rho, and the imaginary component as
// angular value theta
template <typename T> struct test_polar {
  bool operator()(sycl::queue &Q, cmplx<T> init, cmplx<T> ref = cmplx<T>(0, 0),
                  bool use_ref = false) {
    bool pass = true;

    auto *cplx_out = sycl::malloc_shared<experimental::complex<T>>(1, Q);

    /*Get std::complex output*/
    std::complex<T> std_out{ref.re, ref.im};
    if (!use_ref)
      std_out = std::polar(init.re, init.im);

    /*Check cplx::complex output from device*/
    Q.single_task([=]() {
       cplx_out[0] = experimental::polar<T>(init.re, init.im);
     }).wait();

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ true);

    /*Check cplx::complex output from host*/
    cplx_out[0] = experimental::polar<T>(init.re, init.im);

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ false);

    sycl::free(cplx_out, Q);

    return pass;
  }
};

int main() {
  sycl::queue Q;

  bool test_passes = true;

  {
    test_cases<test_acos> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_asin> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_atan> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_acosh> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_asinh> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_atanh> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_conj> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_cos> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_cosh> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_log> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_log10> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_proj> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_sin> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_sinh> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_sqrt> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_tan> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_tanh> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_abs> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_arg> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_norm> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_polar> test;
    test_passes &= test(Q);
  }

  return !test_passes;
}
