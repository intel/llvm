// This test checks edge cases handling for std::exp(std::complex<double>) used
// in SYCL kernels.
//
// REQUIRES: aspect-fp64
// UNSUPPORTED: hip || cuda
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <cmath>
#include <complex>
#include <set>

// To simplify maintanence of those comments specifying indexes of test cases
// in the array below, please add new test cases at the end of the list.
constexpr std::complex<double> testcases[] = {
    /* 0 */ std::complex<double>(1.e-6, 1.e-6),
    /* 1 */ std::complex<double>(-1.e-6, 1.e-6),
    /* 2 */ std::complex<double>(-1.e-6, -1.e-6),
    /* 3 */ std::complex<double>(1.e-6, -1.e-6),

    /* 4 */ std::complex<double>(1.e+6, 1.e-6),
    /* 5 */ std::complex<double>(-1.e+6, 1.e-6),
    /* 6 */ std::complex<double>(-1.e+6, -1.e-6),
    /* 7 */ std::complex<double>(1.e+6, -1.e-6),

    /* 8 */ std::complex<double>(1.e-6, 1.e+6),
    /* 9 */ std::complex<double>(-1.e-6, 1.e+6),
    /* 10 */ std::complex<double>(-1.e-6, -1.e+6),
    /* 11 */ std::complex<double>(1.e-6, -1.e+6),

    /* 12 */ std::complex<double>(1.e+6, 1.e+6),
    /* 13 */ std::complex<double>(-1.e+6, 1.e+6),
    /* 14 */ std::complex<double>(-1.e+6, -1.e+6),
    /* 15 */ std::complex<double>(1.e+6, -1.e+6),

    /* 16 */ std::complex<double>(-0, -1.e-6),
    /* 17 */ std::complex<double>(-0, 1.e-6),
    /* 18 */ std::complex<double>(-0, 1.e+6),
    /* 19 */ std::complex<double>(-0, -1.e+6),
    /* 20 */ std::complex<double>(0, -1.e-6),
    /* 21 */ std::complex<double>(0, 1.e-6),
    /* 22 */ std::complex<double>(0, 1.e+6),
    /* 23 */ std::complex<double>(0, -1.e+6),

    /* 24 */ std::complex<double>(-1.e-6, -0),
    /* 25 */ std::complex<double>(1.e-6, -0),
    /* 26 */ std::complex<double>(1.e+6, -0),
    /* 27 */ std::complex<double>(-1.e+6, -0),
    /* 28 */ std::complex<double>(-1.e-6, 0),
    /* 29 */ std::complex<double>(1.e-6, 0),
    /* 30 */ std::complex<double>(1.e+6, 0),
    /* 31 */ std::complex<double>(-1.e+6, 0),

    /* 32 */ std::complex<double>(NAN, NAN),
    /* 33 */ std::complex<double>(-INFINITY, NAN),
    /* 34 */ std::complex<double>(-2, NAN),
    /* 35 */ std::complex<double>(-1, NAN),
    /* 36 */ std::complex<double>(-0.5, NAN),
    /* 37 */ std::complex<double>(-0., NAN),
    /* 38 */ std::complex<double>(+0., NAN),
    /* 39 */ std::complex<double>(0.5, NAN),
    /* 40 */ std::complex<double>(1, NAN),
    /* 41 */ std::complex<double>(2, NAN),
    /* 42 */ std::complex<double>(INFINITY, NAN),

    /* 43 */ std::complex<double>(NAN, -INFINITY),
    /* 44 */ std::complex<double>(-INFINITY, -INFINITY),
    /* 45 */ std::complex<double>(-2, -INFINITY),
    /* 46 */ std::complex<double>(-1, -INFINITY),
    /* 47 */ std::complex<double>(-0.5, -INFINITY),
    /* 48 */ std::complex<double>(-0., -INFINITY),
    /* 49 */ std::complex<double>(+0., -INFINITY),
    /* 50 */ std::complex<double>(0.5, -INFINITY),
    /* 51 */ std::complex<double>(1, -INFINITY),
    /* 52 */ std::complex<double>(2, -INFINITY),
    /* 53 */ std::complex<double>(INFINITY, -INFINITY),

    /* 54 */ std::complex<double>(NAN, -2),
    /* 55 */ std::complex<double>(-INFINITY, -2),
    /* 56 */ std::complex<double>(-2, -2),
    /* 57 */ std::complex<double>(-1, -2),
    /* 58 */ std::complex<double>(-0.5, -2),
    /* 59 */ std::complex<double>(-0., -2),
    /* 60 */ std::complex<double>(+0., -2),
    /* 61 */ std::complex<double>(0.5, -2),
    /* 62 */ std::complex<double>(1, -2),
    /* 63 */ std::complex<double>(2, -2),
    /* 64 */ std::complex<double>(INFINITY, -2),

    /* 65 */ std::complex<double>(NAN, -1),
    /* 66 */ std::complex<double>(-INFINITY, -1),
    /* 67 */ std::complex<double>(-2, -1),
    /* 68 */ std::complex<double>(-1, -1),
    /* 69 */ std::complex<double>(-0.5, -1),
    /* 70 */ std::complex<double>(-0., -1),
    /* 71 */ std::complex<double>(+0., -1),
    /* 72 */ std::complex<double>(0.5, -1),
    /* 73 */ std::complex<double>(1, -1),
    /* 74 */ std::complex<double>(2, -1),
    /* 75 */ std::complex<double>(INFINITY, -1),

    /* 76 */ std::complex<double>(NAN, -0.5),
    /* 77 */ std::complex<double>(-INFINITY, -0.5),
    /* 78 */ std::complex<double>(-2, -0.5),
    /* 79 */ std::complex<double>(-1, -0.5),
    /* 80 */ std::complex<double>(-0.5, -0.5),
    /* 81 */ std::complex<double>(-0., -0.5),
    /* 82 */ std::complex<double>(+0., -0.5),
    /* 83 */ std::complex<double>(0.5, -0.5),
    /* 84 */ std::complex<double>(1, -0.5),
    /* 85 */ std::complex<double>(2, -0.5),
    /* 86 */ std::complex<double>(INFINITY, -0.5),

    /* 87 */ std::complex<double>(NAN, -0.),
    /* 88 */ std::complex<double>(-INFINITY, -0.),
    /* 89 */ std::complex<double>(-2, -0.),
    /* 90 */ std::complex<double>(-1, -0.),
    /* 91 */ std::complex<double>(-0.5, -0.),
    /* 92 */ std::complex<double>(-0., -0.),
    /* 93 */ std::complex<double>(+0., -0.),
    /* 94 */ std::complex<double>(0.5, -0.),
    /* 95 */ std::complex<double>(1, -0.),
    /* 96 */ std::complex<double>(2, -0.),
    /* 97 */ std::complex<double>(INFINITY, -0.),

    /* 98 */ std::complex<double>(NAN, +0.),
    /* 99 */ std::complex<double>(-INFINITY, +0.),
    /* 100 */ std::complex<double>(-2, +0.),
    /* 101 */ std::complex<double>(-1, +0.),
    /* 102 */ std::complex<double>(-0.5, +0.),
    /* 103 */ std::complex<double>(-0., +0.),
    /* 104 */ std::complex<double>(+0., +0.),
    /* 105 */ std::complex<double>(0.5, +0.),
    /* 106 */ std::complex<double>(1, +0.),
    /* 107 */ std::complex<double>(2, +0.),
    /* 108 */ std::complex<double>(INFINITY, +0.),

    /* 109 */ std::complex<double>(NAN, 0.5),
    /* 110 */ std::complex<double>(-INFINITY, 0.5),
    /* 111 */ std::complex<double>(-2, 0.5),
    /* 112 */ std::complex<double>(-1, 0.5),
    /* 113 */ std::complex<double>(-0.5, 0.5),
    /* 114 */ std::complex<double>(-0., 0.5),
    /* 115 */ std::complex<double>(+0., 0.5),
    /* 116 */ std::complex<double>(0.5, 0.5),
    /* 117 */ std::complex<double>(1, 0.5),
    /* 118 */ std::complex<double>(2, 0.5),
    /* 119 */ std::complex<double>(INFINITY, 0.5),

    /* 120 */ std::complex<double>(NAN, 1),
    /* 121 */ std::complex<double>(-INFINITY, 1),
    /* 122 */ std::complex<double>(-2, 1),
    /* 123 */ std::complex<double>(-1, 1),
    /* 124 */ std::complex<double>(-0.5, 1),
    /* 125 */ std::complex<double>(-0., 1),
    /* 126 */ std::complex<double>(+0., 1),
    /* 127 */ std::complex<double>(0.5, 1),
    /* 128 */ std::complex<double>(1, 1),
    /* 129 */ std::complex<double>(2, 1),
    /* 130 */ std::complex<double>(INFINITY, 1),

    /* 131 */ std::complex<double>(NAN, 2),
    /* 132 */ std::complex<double>(-INFINITY, 2),
    /* 133 */ std::complex<double>(-2, 2),
    /* 134 */ std::complex<double>(-1, 2),
    /* 135 */ std::complex<double>(-0.5, 2),
    /* 136 */ std::complex<double>(-0., 2),
    /* 137 */ std::complex<double>(+0., 2),
    /* 138 */ std::complex<double>(0.5, 2),
    /* 139 */ std::complex<double>(1, 2),
    /* 140 */ std::complex<double>(2, 2),
    /* 141 */ std::complex<double>(INFINITY, 2),

    /* 142 */ std::complex<double>(NAN, INFINITY),
    /* 143 */ std::complex<double>(-INFINITY, INFINITY),
    /* 144 */ std::complex<double>(-2, INFINITY),
    /* 145 */ std::complex<double>(-1, INFINITY),
    /* 146 */ std::complex<double>(-0.5, INFINITY),
    /* 147 */ std::complex<double>(-0., INFINITY),
    /* 148 */ std::complex<double>(+0., INFINITY),
    /* 149 */ std::complex<double>(0.5, INFINITY),
    /* 150 */ std::complex<double>(1, INFINITY),
    /* 151 */ std::complex<double>(2, INFINITY),
    /* 152 */ std::complex<double>(INFINITY, INFINITY)};

bool check(bool cond, const std::string &cond_str, int line, unsigned testcase,
           const std::set<unsigned> &known_fails) {
  if (!cond && !known_fails.count(testcase)) {
    std::cout << "Assertion " << cond_str << " (line " << line
              << ") failed for testcase #" << testcase << std::endl;
    return false;
  } else if (cond && known_fails.count(testcase)) {
    std::cout << "Assertion " << cond_str << " (line " << line
              << ") passed for testcase #" << testcase << std::endl;
    std::cout << "However, it was recorded as a known failure and therefore "
                 "the test needs to be updated"
              << std::endl;
    return false;
  }
  return true;
}

int main() try {
  sycl::queue q;

  constexpr unsigned N = sizeof(testcases) / sizeof(testcases[0]);

  sycl::buffer<std::complex<double>> results(sycl::range{N});

  q.submit([&](sycl::handler &cgh) {
     sycl::accessor acc(results, cgh, sycl::write_only);
     cgh.parallel_for(sycl::range{N}, [=](sycl::item<1> it) {
       acc[it] = std::exp(testcases[it]);
     });
   }).wait_and_throw();

  // FIXME: the set below should be empty and therefore removed
  std::set<unsigned> known_fails = {32,  34,  35,  36,  37,  38,  39,  40,  41,
                                    43,  45,  46,  47,  48,  49,  50,  51,  52,
                                    54,  65,  76,  109, 120, 131, 142, 144, 145,
                                    146, 147, 148, 149, 150, 151};

  bool passed = true;

  // Note: this macro is expected to be used within a loop
#define CHECK(cond, pass_marker, ...)                                          \
  if (!check((cond), #cond, __LINE__, __VA_ARGS__)) {                          \
    pass_marker = false;                                                       \
    continue;                                                                  \
  }

  // Based on https://en.cppreference.com/w/cpp/numeric/complex/exp
  // z below refers to the argument passed to std::exp(complex<double>)
  sycl::host_accessor acc(results);
  for (unsigned i = 0; i < N; ++i) {
    std::complex<double> r = acc[i];
    // If z is (+/-0, +0), the result is (1, +0)
    if (testcases[i].real() == 0 && testcases[i].imag() == 0) {
      CHECK(r.real() == 1.0, passed, i, known_fails);
      CHECK(r.imag() == 0, passed, i, known_fails);
      CHECK(std::signbit(testcases[i].imag()) == std::signbit(r.imag()), passed,
            i, known_fails);
      // If z is (x, +inf) (for any finite x), the result is (NaN, NaN)
    } else if (std::isfinite(testcases[i].real()) &&
               std::isinf(testcases[i].imag())) {
      CHECK(std::isnan(r.real()), passed, i, known_fails);
      CHECK(std::isnan(r.imag()), passed, i, known_fails);
      // If z is (x, NaN) (for any finite x), the result is (NaN, NaN)
    } else if (std::isfinite(testcases[i].real()) &&
               std::isnan(testcases[i].imag())) {
      CHECK(std::isnan(r.real()), passed, i, known_fails);
      CHECK(std::isnan(r.imag()), passed, i, known_fails);
      // If z is (+inf, +0), the result is (+inf, +0)
    } else if (std::isinf(testcases[i].real()) && testcases[i].real() > 0 &&
               testcases[i].imag() == 0) {
      CHECK(std::isinf(r.real()), passed, i, known_fails);
      CHECK(r.real() > 0, passed, i, known_fails);
      CHECK(r.imag() == 0, passed, i, known_fails);
      CHECK(std::signbit(testcases[i].imag()) == std::signbit(r.imag()), passed,
            i, known_fails);
      // If z is (-inf, +inf), the result is (+/-0, +/-0) (signs are
      // unspecified)
    } else if (std::isinf(testcases[i].real()) && testcases[i].real() < 0 &&
               std::isinf(testcases[i].imag())) {
      CHECK(r.real() == 0, passed, i, known_fails);
      CHECK(r.imag() == 0, passed, i, known_fails);
      // If z is (+inf, +inf), the result is (+/-inf, NaN), (the sign of the
      // real part is unspecified)
    } else if (std::isinf(testcases[i].real()) && testcases[i].real() > 0 &&
               std::isinf(testcases[i].imag())) {
      CHECK(std::isinf(r.real()), passed, i, known_fails);
      CHECK(std::isnan(r.imag()), passed, i, known_fails);
      // If z is (-inf, NaN), the result is (+/-0, +/-0) (signs are unspecified)
    } else if (std::isinf(testcases[i].real()) && testcases[i].real() < 0 &&
               std::isnan(testcases[i].imag())) {
      CHECK(r.real() == 0, passed, i, known_fails);
      CHECK(r.imag() == 0, passed, i, known_fails);
      // If z is (+inf, NaN), the result is (+/-inf, NaN) (the sign of the real
      // part is unspecified)
    } else if (std::isinf(testcases[i].real()) && testcases[i].real() > 0 &&
               std::isnan(testcases[i].imag())) {
      CHECK(std::isinf(r.real()), passed, i, known_fails);
      CHECK(std::isnan(r.imag()), passed, i, known_fails);
      // If z is (NaN, +0), the result is (NaN, +0)
    } else if (std::isnan(testcases[i].real()) && testcases[i].imag() == 0) {
      CHECK(std::isnan(r.real()), passed, i, known_fails);
      CHECK(r.imag() == 0, passed, i, known_fails);
      CHECK(std::signbit(testcases[i].imag()) == std::signbit(r.imag()), passed,
            i, known_fails);
      // If z is (NaN, y) (for any nonzero y), the result is (NaN,NaN)
    } else if (std::isnan(testcases[i].real()) && testcases[i].imag() != 0) {
      CHECK(std::isnan(r.real()), passed, i, known_fails);
      CHECK(std::isnan(r.imag()), passed, i, known_fails);
      // If z is (NaN, NaN), the result is (NaN, NaN)
    } else if (std::isnan(testcases[i].real()) &&
               std::isnan(testcases[i].imag())) {
      CHECK(std::isnan(r.real()), passed, i, known_fails);
      CHECK(std::isnan(r.imag()), passed, i, known_fails);
      // Those tests were taken from oneDPL, not sure what is the corner case
      // they are covering here
    } else if (std::isfinite(testcases[i].imag()) &&
               std::abs(testcases[i].imag()) <= 1) {
      CHECK(!std::signbit(r.real()), passed, i, known_fails);
      CHECK(std::signbit(r.imag()) == std::signbit(testcases[i].imag()), passed,
            i, known_fails);
      // Those tests were taken from oneDPL, not sure what is the corner case
      // they are covering here
    } else if (std::isinf(r.real()) && testcases[i].imag() == 0) {
      CHECK(r.imag() == 0, passed, i, known_fails);
      CHECK(std::signbit(testcases[i].imag()) == std::signbit(r.imag()), passed,
            i, known_fails);
    }
    // FIXME: do we have the following cases covered?
    // If z is (-inf, y) (for any finite y), the result is +0 cis(y)
    // If z is (+inf, y) (for any finite nonzero y), the result is +inf cis(y)
  }

  return passed ? 0 : 1;
} catch (sycl::exception &e) {
  std::cout << "Caught sync sycl exception: " << e.what() << std::endl;
  return 2;
}
