// UNSUPPORTED: target-native_cpu
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20142
// UNSUPPORTED: windows && !aspect-fp64
// UNSUPPORTED-INTENDED: On Windows std::complex<float> device math comes from
// MSVC's STL, which calls double C math functions (see std_complex_math_test).

// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}
// RUN: %{build} %{mathflags} -o %t.out
// RUN: %{run} %t.out

// std::acos/asin/acosh/asinh on device for large arguments with negative real
// part (positive imaginary part for asin): z + sqrt(z^2 -+ 1) cancelled to zero
// and the result was -inf. Compare against the host implementation.

#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

template <typename T> int run(queue &q, const std::complex<T> *in, int n) {
  using C = std::complex<T>;
  auto *d_in = malloc_shared<C>(n, q);
  auto *d_out = malloc_shared<C>(4 * n, q);
  for (int i = 0; i < n; ++i)
    d_in[i] = in[i];

  q.parallel_for(range<1>(n), [=](id<1> i) {
    const size_t idx = i[0];
    d_out[4 * idx + 0] = std::acos(d_in[idx]);
    d_out[4 * idx + 1] = std::asin(d_in[idx]);
    d_out[4 * idx + 2] = std::acosh(d_in[idx]);
    d_out[4 * idx + 3] = std::asinh(d_in[idx]);
  });
  q.wait_and_throw();

  const char *names[] = {"acos", "asin", "acosh", "asinh"};
  const T tol = 8 * std::numeric_limits<T>::epsilon();
  int fails = 0;
  for (int i = 0; i < n; ++i) {
    C ref[] = {std::acos(in[i]), std::asin(in[i]), std::acosh(in[i]),
               std::asinh(in[i])};
    for (int f = 0; f < 4; ++f) {
      C got = d_out[4 * i + f];
      bool inf = std::isinf(got.real()) || std::isinf(got.imag());
      if (inf || std::abs(got - ref[f]) > tol * std::abs(ref[f])) {
        std::cout << names[f] << "(" << in[i] << ") = " << got << ", expected "
                  << ref[f] << "\n";
        ++fails;
      }
    }
  }
  free(d_in, q);
  free(d_out, q);
  return fails;
}

int main() try {
  queue q{[](sycl::exception_list el) {
    for (auto &e : el)
      std::rethrow_exception(e);
  }};
  int fails = 0;

  {
    using C = std::complex<float>;
    const float big = 1e5f;
    C in[] = {C(-4e3f, 1),   C(-1e4f, 1),   C(-1e6f, 1), C(-big, -1),
              C(1, big),     C(-1, big),    C(-1, -big), C(-0.0f, big),
              C(-big, 0.0f), C(-big, -0.0f)};
    fails += run<float>(q, in, std::size(in));
  }

  if (q.get_device().has(aspect::fp64)) {
    using C = std::complex<double>;
    const double big = 1e8;
    C in[] = {C(-4e7, 1),   C(-9e7, 1),   C(-4.45712982e8, 1), C(-4.5e15, 1),
              C(-big, -1),  C(1, big),    C(-1, big),          C(-1, -big),
              C(-0.0, big), C(-big, 0.0), C(-big, -0.0)};
    fails += run<double>(q, in, std::size(in));
  }

  return fails;
} catch (const std::exception &e) {
  std::cout << "exception: " << e.what() << "\n";
  return 1;
}
