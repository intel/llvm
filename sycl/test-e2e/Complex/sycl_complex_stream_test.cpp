// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-finite-math-only%} %else %{-fno-finite-math-only%}

// RUN: %{build} -fsycl-device-code-split=per_kernel %{mathflags} -o %t.out
// RUN: %{run} %t.out

#include "sycl_complex_helper.hpp"

template <typename T> struct test_sycl_stream_operator {
  bool operator()(sycl::queue &Q, cmplx<T> init) {
    sycl::buffer<experimental::complex<T>> cplx_out_buf{sycl::range{1}};
    sycl::host_accessor cplx_out_host{cplx_out_buf};
    cplx_out_host[0] = experimental::complex<T>(init.re, init.im);
    Q.submit([&](sycl::handler &CGH) {
       sycl::accessor cplx_out{cplx_out_buf, CGH};
       sycl::stream Out(512, 20, CGH);
       CGH.parallel_for<>(sycl::range<1>(1), [=](sycl::id<1> idx) {
         Out << cplx_out[idx] << sycl::endl;
       });
     }).wait();
    return true;
  }
};

// Host only tests for std::basic_ostream and std::basic_istream
template <typename T> struct test_ostream_operator {
  bool operator()(cmplx<T> init) {
    experimental::complex<T> c(init.re, init.im);

    std::ostringstream os;
    os << c;

    std::ostringstream ref_oss;
    ref_oss << "(" << init.re << "," << init.im << ")";

    if (ref_oss.str() == os.str())
      return true;
    return false;
  }
};

template <typename T> struct test_istream_operator {
  bool operator()(cmplx<T> init) {
    experimental::complex<T> c(init.re, init.im);

    std::ostringstream ref_oss;
    ref_oss << "(" << init.re << "," << init.im << ")";

    std::istringstream iss(ref_oss.str());

    iss >> c;

    return check_results(c, std::complex<T>(init.re, init.im),
                         /*is_device*/ false);
  }
};

template <typename T> bool test_common(sycl::queue Q) {
  bool test_passes = true;
  test_passes &=
      test_valid_types<test_sycl_stream_operator>(Q, cmplx<T>(1.5, -1.0));
  test_passes &= test_valid_types<test_sycl_stream_operator>(
      Q, cmplx<T>(INFINITY, INFINITY));
  test_passes &=
      test_valid_types<test_sycl_stream_operator>(Q, cmplx<T>(NAN, NAN));

  test_passes &= test_valid_types<test_ostream_operator>(cmplx<T>(1.5, -1.0));
  test_passes &=
      test_valid_types<test_ostream_operator>(cmplx<T>(INFINITY, INFINITY));
  test_passes &= test_valid_types<test_ostream_operator>(cmplx<T>(NAN, NAN));

  test_passes &= test_valid_types<test_istream_operator>(cmplx<T>(1.5, -1.0));
  test_passes &=
      test_valid_types<test_istream_operator>(cmplx<T>(INFINITY, INFINITY));
  test_passes &= test_valid_types<test_istream_operator>(cmplx<T>(NAN, NAN));
  return test_passes;
}

int main() {
  sycl::queue Q;

  bool test_passes = true;

  test_passes &= test_common<float>(Q);
  if (Q.get_device().has(sycl::aspect::fp16))
    test_passes &= test_common<sycl::half>(Q);
  if (Q.get_device().has(sycl::aspect::fp64))
    test_passes &= test_common<double>(Q);

  if (!test_passes)
    std::cerr << "Stream operator with complex test fails\n";

  return !test_passes;
}
