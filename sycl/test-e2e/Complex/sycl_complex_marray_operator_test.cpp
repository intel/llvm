// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "sycl_complex_marray_test_cases.hpp"

#define TEST_BASIC_OPERATOR(op_name, op)                                       \
  template <typename T> struct test_##op_name {                                \
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
          std_out[i] = std_in[i] op std_in[i];                                 \
        }                                                                      \
      }                                                                        \
                                                                               \
      Q.submit([&](sycl::handler &cgh) {                                       \
        sycl::accessor cplx_out{cplx_out_buf, cgh};                            \
                                                                               \
        cgh.single_task([=]() { cplx_out[0] = cplx_in op cplx_in; });          \
      });                                                                      \
                                                                               \
      /* Check cplx::complex output from device */                             \
      sycl::host_accessor cplx_out_acc{cplx_out_buf};                          \
      pass &= check_results(cplx_out_acc[0], convert_marray<T>(std_out),       \
                            /*is_device*/ true);                               \
                                                                               \
      /* Check cplx::complex output from host */                               \
      cplx_out_acc[0] = cplx_in op cplx_in;                                    \
      pass &= check_results(cplx_out_acc[0], convert_marray<T>(std_out),       \
                            /*is_device*/ false);                              \
                                                                               \
      return pass;                                                             \
    }                                                                          \
  };

TEST_BASIC_OPERATOR(add, +)
TEST_BASIC_OPERATOR(sub, -)
TEST_BASIC_OPERATOR(mul, *)
TEST_BASIC_OPERATOR(div, /)

#undef TEST_BASIC_OPERATOR

#define TEST_ASSIGN_OPERATOR(op_name, op)                                      \
  template <typename T> struct test_##op_name {                                \
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
      sycl::marray<std::complex<X>, DEFAULT_TEST_CASE_SIZE> std_out{std_in};   \
      sycl::buffer<                                                            \
          sycl::marray<experimental::complex<T>, DEFAULT_TEST_CASE_SIZE>>      \
          cplx_out_buf{&cplx_in, sycl::range{1}};                              \
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
          std_out[i] op std_in[i];                                             \
        }                                                                      \
      }                                                                        \
                                                                               \
      Q.submit([&](sycl::handler &cgh) {                                       \
        sycl::accessor cplx_out{cplx_out_buf, cgh};                            \
                                                                               \
        cgh.single_task([=]() { cplx_out[0] op cplx_in; });                    \
      });                                                                      \
                                                                               \
      /* Check cplx::complex output from device */                             \
      sycl::host_accessor cplx_out_acc{cplx_out_buf};                          \
      pass &= check_results(cplx_out_acc[0], convert_marray<T>(std_out),       \
                            /*is_device*/ true);                               \
                                                                               \
      for (std::size_t i = 0; i < DEFAULT_TEST_CASE_SIZE; ++i) {               \
        cplx_in[i] = experimental::complex<T>{static_cast<T>(init[i].re),      \
                                              static_cast<T>(init[i].im)};     \
      }                                                                        \
                                                                               \
      /* Check cplx::complex output from host */                               \
      cplx_out_acc[0] op cplx_in;                                              \
      pass &= check_results(cplx_out_acc[0], convert_marray<T>(std_out),       \
                            /*is_device*/ false);                              \
                                                                               \
      return pass;                                                             \
    }                                                                          \
  };

TEST_ASSIGN_OPERATOR(assign_add, +=)
TEST_ASSIGN_OPERATOR(assign_sub, -=)
TEST_ASSIGN_OPERATOR(assign_mul, *=)
TEST_ASSIGN_OPERATOR(assign_div, /=)

#undef TEST_ASSIGN_OPERATOR

#define TEST_UNARY_OPERATOR(op_name, op)                                       \
  template <typename T> struct test_##op_name {                                \
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
          std_out[i] = op std_in[i];                                           \
        }                                                                      \
      }                                                                        \
                                                                               \
      Q.submit([&](sycl::handler &cgh) {                                       \
        sycl::accessor cplx_out{cplx_out_buf, cgh};                            \
                                                                               \
        cgh.single_task([=]() { cplx_out[0] = op cplx_in; });                  \
      });                                                                      \
                                                                               \
      /* Check cplx::complex output from device */                             \
      sycl::host_accessor cplx_out_acc{cplx_out_buf};                          \
      pass &= check_results(cplx_out_acc[0], convert_marray<T>(std_out),       \
                            /*is_device*/ true);                               \
                                                                               \
      /* Check cplx::complex output from host */                               \
      cplx_out_acc[0] = op cplx_in;                                            \
      pass &= check_results(cplx_out_acc[0], convert_marray<T>(std_out),       \
                            /*is_device*/ false);                              \
                                                                               \
      return pass;                                                             \
    }                                                                          \
  };

TEST_UNARY_OPERATOR(unary_add, +)
TEST_UNARY_OPERATOR(unary_sub, -)

#undef TEST_UNARY_OPERATOR

int main() {
  sycl::queue Q;

  bool test_passes = true;

  /* Test basic arithmetic operator */

  {
    marray_cplx_test_cases<test_add> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_sub> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_mul> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_div> test;
    test_passes &= test(Q);
  }

  /* Test assign arithmetic operator */

  {
    marray_cplx_test_cases<test_assign_add> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_assign_sub> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_assign_mul> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_assign_div> test;
    test_passes &= test(Q);
  }

  /* Test unary operator */

  {
    marray_cplx_test_cases<test_unary_add> test;
    test_passes &= test(Q);
  }

  {
    marray_cplx_test_cases<test_unary_sub> test;
    test_passes &= test(Q);
  }

  return !test_passes;
}
