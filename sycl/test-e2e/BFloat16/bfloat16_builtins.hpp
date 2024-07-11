#pragma once

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/bfloat16_math.hpp>

#include <cmath>
#include <vector>

using namespace sycl;
using namespace sycl::ext::oneapi;
using namespace sycl::ext::oneapi::experimental;

constexpr int N = 60; // divisible by all tested array sizes
constexpr float bf16_eps = 0.00390625;

float make_fp32(uint16_t x) {
  uint32_t y = x;
  y = y << 16;
  return sycl::bit_cast<float>(y);
}

bool check(float a, float b) {
  return sycl::fabs(2 * (a - b) / (a + b)) > bf16_eps * 2;
}

bool check(bool a, bool b) { return (a != b); }

void test() {
  queue q;

  std::vector<float> a(N), b(N), c(N);
  int err = 0;

  for (int i = 0; i < N; i++) {
    a[i] = (i - N / 2) / (float)N;
    b[i] = (N / 2 - i) / (float)N;
    c[i] = (float)(3 * i);
  }

  auto test = [&](auto ExpFunc, auto RefFunc, auto NumOperands) {
    static_assert(NumOperands >= 1 && NumOperands <= 3);
    {
      buffer<float> a_buf(&a[0], N);
      buffer<float> b_buf(&b[0], N);
      buffer<float> c_buf(&c[0], N);
      buffer<int> err_buf(&err, 1);
      q.submit([&](handler &cgh) {
        accessor A(a_buf, cgh);
        accessor B(b_buf, cgh);
        accessor C(c_buf, cgh);
        accessor ERR(err_buf, cgh);
        cgh.parallel_for(N, [=](id<1> index) {
          auto ExpArg = [&](auto acc) { return bfloat16{acc[index]}; };
          auto RefArg = [&](auto acc) { return float{bfloat16{acc[index]}}; };

          bool failure = false;
          if constexpr (NumOperands == 1) {
            failure |= check(ExpFunc(ExpArg(A)), RefFunc(RefArg(A)));
          } else if constexpr (NumOperands == 2) {
            failure |= check(ExpFunc(ExpArg(A), ExpArg(B)),
                             RefFunc(RefArg(A), RefArg(B)));
          } else if constexpr (NumOperands == 3) {
            failure |= check(ExpFunc(ExpArg(A), ExpArg(B), ExpArg(C)),
                             RefFunc(RefArg(A), RefArg(B), RefArg(C)));
          }

          if (failure)
            ERR[0] = 1;
        });
      });
    }
    assert(err == 0);

    sycl::detail::loop<5>([&](auto SZ_MINUS_ONE) {
      constexpr int SZ = SZ_MINUS_ONE + 1;
      {
        buffer<float, 2> a_buf{&a[0], range<2>{N / SZ, SZ}};
        buffer<float, 2> b_buf{&b[0], range<2>{N / SZ, SZ}};
        buffer<float, 2> c_buf{&c[0], range<2>{N / SZ, SZ}};
        buffer<int> err_buf(&err, 1);
        q.submit([&](handler &cgh) {
          accessor A(a_buf, cgh);
          accessor B(b_buf, cgh);
          accessor C(c_buf, cgh);
          accessor ERR(err_buf, cgh);
          cgh.parallel_for(N / SZ, [=](id<1> index) {
            marray<bfloat16, SZ> arg0, arg1, arg2;
            for (int i = 0; i < SZ; i++) {
              arg0[i] = A[index][i];
              arg1[i] = B[index][i];
              arg2[i] = C[index][i];
            }
            auto res = [&]() {
              if constexpr (NumOperands == 1) {
                return ExpFunc(arg0);
              } else if constexpr (NumOperands == 2) {
                return ExpFunc(arg0, arg1);
              } else if constexpr (NumOperands == 3) {
                return ExpFunc(arg0, arg1, arg2);
              }
            }();

            bool failure = false;
            for (int i = 0; i < SZ; ++i) {
              auto RefArg = [&](auto acc) {
                return float{bfloat16{acc[index][i]}};
              };
              if constexpr (NumOperands == 1) {
                failure |= check(res[i], RefFunc(RefArg(A)));
              } else if constexpr (NumOperands == 2) {
                failure |= check(res[i], RefFunc(RefArg(A), RefArg(B)));
              } else if constexpr (NumOperands == 3) {
                failure |=
                    check(res[i], RefFunc(RefArg(A), RefArg(B), RefArg(C)));
              }
            }
            if (failure)
              ERR[0] = 1;
          });
        });
      }
      assert(err == 0);
    });
  };

#define TEST(NAME, NUM_OPERANDS)                                               \
  test(                                                                        \
      [](auto... args) {                                                       \
        return sycl::ext::oneapi::experimental::NAME(args...);                 \
      },                                                                       \
      [](auto... args) { return sycl::NAME(args...); },                        \
      std::integral_constant<int, NUM_OPERANDS>{})

  TEST(fabs, 1);

  TEST(fmin, 2);
  TEST(fmax, 2);
  TEST(fma, 3);

  auto test_nan = [&](auto ExpFunc) {
    float check_nan = 0;
    {
      buffer<int> err_buf(&err, 1);
      buffer<float> nan_buf(&check_nan, 1);
      q.submit([&](handler &cgh) {
        accessor ERR(err_buf, cgh);
        accessor checkNAN(nan_buf, cgh);
        cgh.single_task([=]() {
          checkNAN[0] = ExpFunc(bfloat16{NAN}, bfloat16{NAN});
          if ((ExpFunc(bfloat16{2}, bfloat16{NAN}) != 2) ||
              (ExpFunc(bfloat16{NAN}, bfloat16{2}) != 2)) {
            ERR[0] = 1;
          }
        });
      });
    }
    assert(err == 0);
    assert(std::isnan(check_nan));
  };
  test_nan([](auto... args) {
    return sycl::ext::oneapi::experimental::fmin(args...);
  });
  test_nan([](auto... args) {
    return sycl::ext::oneapi::experimental::fmax(args...);
  });

  // Insert NAN value in a to test isnan
  a[0] = a[N - 1] = NAN;
  TEST(isnan, 1);

  // Orignal input 'a[0...N-1]' are in range [-0.5, 0.5),
  // need to update it for generic math testing.
  // sin, cos testing
  for (int i = 0; i < N; ++i) {
    a[i] = (i / (float)(N - 1)) * 6.28;
    if ((i & 0x1) == 0x1)
      a[i] = -a[i];
  }
  TEST(cos, 1);
  TEST(sin, 1);

  // ceil, floor, trunc, exp, exp2, exp10, rint testing
  TEST(ceil, 1);
  TEST(floor, 1);
  TEST(trunc, 1);
  TEST(exp, 1);
  TEST(exp10, 1);
  TEST(exp2, 1);
  TEST(rint, 1);

  // log, log2, log10, sqrt, rsqrt testing, the input
  // must be positive.
  for (int i = 0; i < N; ++i)
    a[i] = a[i] + 8.5;

  TEST(sqrt, 1);
  TEST(rsqrt, 1);
  TEST(log, 1);
  TEST(log2, 1);
  TEST(log10, 1);
}
