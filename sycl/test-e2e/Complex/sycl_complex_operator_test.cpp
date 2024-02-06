// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

#include "sycl_complex_helper.hpp"
#include "sycl_complex_operator_test_cases.hpp"

#define test_op(name, op)                                                      \
  template <typename T> struct name {                                          \
    bool operator()(sycl::queue &Q, T init_re1, T init_im1, T init_re2,        \
                    T init_im2) {                                              \
      bool pass = true;                                                        \
                                                                               \
      auto std_in1 = init_std_complex(init_re1, init_im1);                     \
      auto std_in2 = init_std_complex(init_re2, init_im2);                     \
      experimental::complex<T> cplx_input1{init_re1, init_im1};                \
      experimental::complex<T> cplx_input2{init_re2, init_im2};                \
                                                                               \
      sycl::buffer<experimental::complex<T>> cplx_out_buf{sycl::range{1}};     \
      std::complex<T> std_out;                                                 \
      std_out = std_in1 op std_in2;                                            \
      Q.submit([&](sycl::handler &h) {                                         \
        sycl::accessor cplx_out{cplx_out_buf, h};                              \
        h.single_task([=]() { cplx_out[0] = cplx_input1 op cplx_input2; });    \
      });                                                                      \
      sycl::host_accessor cplx_out_acc{cplx_out_buf};                          \
      pass &= check_results(cplx_out_acc[0], std_out, /*is_device*/ true);     \
                                                                               \
      cplx_out_acc[0] = cplx_input1 op cplx_input2;                            \
                                                                               \
      pass &= check_results(cplx_out_acc[0], std_out, /*is_device*/ false);    \
      return pass;                                                             \
    }                                                                          \
  };

test_op(test_add, +);
test_op(test_sub, -);
test_op(test_mul, *);
test_op(test_div, /);

#undef test_op

#define test_op_assign(name, op_assign)                                        \
  template <typename T> struct name {                                          \
    bool operator()(sycl::queue &Q, T init_re1, T init_im1, T init_re2,        \
                    T init_im2) {                                              \
      bool pass = true;                                                        \
                                                                               \
      auto std_in = init_std_complex(init_re1, init_im1);                      \
      experimental::complex<T> cplx_input{init_re1, init_im1};                 \
                                                                               \
      auto std_inout = init_std_complex(init_re2, init_im2);                   \
      experimental::complex<T> cplx;                                           \
      cplx.real(init_re2);                                                     \
      cplx.imag(init_im2);                                                     \
      sycl::buffer<experimental::complex<T>> cplx_out_buf{&cplx,               \
                                                          sycl::range{1}};     \
      std_inout op_assign std_in;                                              \
      Q.submit([&](sycl::handler &h) {                                         \
        sycl::accessor cplx_out{cplx_out_buf, h};                              \
        h.single_task([=]() { cplx_out[0] op_assign cplx_input; });            \
      });                                                                      \
      sycl::host_accessor cplx_out_acc{cplx_out_buf};                          \
                                                                               \
      pass &=                                                                  \
          check_results(cplx_out_acc[0],                                       \
                        std::complex<T>(std_inout.real(), std_inout.imag()),   \
                        /*is_device*/ true);                                   \
                                                                               \
      cplx_out_acc[0].real(init_re2);                                          \
      cplx_out_acc[0].imag(init_im2);                                          \
                                                                               \
      cplx_out_acc[0] op_assign cplx_input;                                    \
                                                                               \
      pass &=                                                                  \
          check_results(cplx_out_acc[0],                                       \
                        std::complex<T>(std_inout.real(), std_inout.imag()),   \
                        /*is_device*/ false);                                  \
                                                                               \
      return pass;                                                             \
    }                                                                          \
  };

test_op_assign(test_add_assign, +=);
test_op_assign(test_sub_assign, -=);
test_op_assign(test_mul_assign, *=);
test_op_assign(test_div_assign, /=);

#undef test_op_assign

// Macro for testing unary operators

#define test_op_unary(name, op)                                                \
  template <typename T> struct name {                                          \
    bool operator()(sycl::queue &Q, T init_re1, T init_im1, T init_re2,        \
                    T init_im2) {                                              \
      bool pass = true;                                                        \
                                                                               \
      auto std_in = init_std_complex(init_re1, init_im1);                      \
      experimental::complex<T> cplx_input{init_re1, init_im1};                 \
                                                                               \
      std::complex<T> std_out{};                                               \
      sycl::buffer<experimental::complex<T>> cplx_out_buf{sycl::range{1}};     \
      std_out = op std_in;                                                     \
      Q.submit([&](sycl::handler &h) {                                         \
        sycl::accessor cplx_out{cplx_out_buf, h};                              \
        h.single_task([=]() { cplx_out[0] op cplx_input; });                   \
      });                                                                      \
      sycl::host_accessor cplx_out_acc{cplx_out_buf};                          \
      pass &= check_results(cplx_out_acc[0], std_out, /*is_device*/ true);     \
                                                                               \
      cplx_out_acc[0] = op cplx_input;                                         \
                                                                               \
      pass &= check_results(cplx_out_acc[0], std_out, /*is_device*/ false);    \
                                                                               \
      return pass;                                                             \
    }                                                                          \
  };

test_op_unary(test_unary_plus, +);
test_op_unary(test_unary_minus, -);

#undef test_op_unary

int main() {
  sycl::queue Q;

  bool test_passes = true;

  {
    test_cases<test_add> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_sub> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_mul> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_div> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_add_assign> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_sub_assign> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_mul_assign> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_div_assign> test;
    test_passes &= test(Q);
  }

  /* Test unary operators */

  {
    test_cases<test_unary_plus> test;
    test_passes &= test(Q);
  }
  {
    test_cases<test_unary_minus> test;
    test_passes &= test(Q);
  }

  return !test_passes;
}
