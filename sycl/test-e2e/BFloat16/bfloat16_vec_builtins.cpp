// RUN: %{build} -fno-fast-math -o %t.out
// RUN: %{run} %t.out

// Test new, ABI-breaking for all platforms.
// RUN:  %if preview-breaking-changes-supported %{  %{build} -fpreview-breaking-changes -o %t-pfrev.out %}
// RUN:  %if preview-breaking-changes-supported %{  %{run} %t-pfrev.out  %}

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/bfloat16_math.hpp>

#include <cmath>
#include <iostream>
#include <vector>

using namespace sycl;
using namespace sycl::ext::oneapi;
using namespace sycl::ext::oneapi::experimental;

constexpr float bf16_eps = 0.00390625;

bool check(float a, float b) {
  return sycl::fabs(2 * (a - b) / (a + b)) > bf16_eps * 2;
}

bool check(bool a, bool b) { return (a != b); }

#define TEST_UNARY_OP(NAME, SZ, RETTY, INPVAL)                                 \
  {                                                                            \
    vec<bfloat16, SZ> arg;                                                     \
    /* Initialize the vector with INPVAL */                                    \
    for (int i = 0; i < SZ; i++) {                                             \
      arg[i] = INPVAL;                                                         \
    }                                                                          \
    /* Perform the operation. */                                               \
    vec<RETTY, SZ> res = sycl::ext::oneapi::experimental::NAME(arg);           \
    vec<RETTY, 2> res2 =                                                       \
        sycl::ext::oneapi::experimental::NAME(arg.template swizzle<0, 0>());   \
    /* Check the result. */                                                    \
    if (res2[0] != res[0] || res2[1] != res[0]) {                              \
      ERR[0] += 1;                                                             \
    }                                                                          \
    for (int i = 0; i < SZ; i++) {                                             \
      if (check(res[i], sycl::NAME(INPVAL))) {                                 \
        ERR[0] += 1;                                                           \
      }                                                                        \
    }                                                                          \
  }

#define TEST_BINARY_OP(NAME, SZ, RETTY, INPVAL)                                \
  {                                                                            \
    vec<bfloat16, SZ> arg, arg2;                                               \
    bfloat16 inpVal2 = 1.0f;                                                   \
    /* Initialize the vector with INPVAL */                                    \
    for (int i = 0; i < SZ; i++) {                                             \
      arg[i] = INPVAL;                                                         \
      arg2[i] = inpVal2;                                                       \
    }                                                                          \
    /* Perform the operation. */                                               \
    vec<RETTY, SZ> res = sycl::ext::oneapi::experimental::NAME(arg, arg2);     \
    /* Swizzle and vec different combination. */                               \
    vec<RETTY, 2> res2 = sycl::ext::oneapi::experimental::NAME(                \
        arg.template swizzle<0, 0>(), arg2.template swizzle<0, 0>());          \
    vec<RETTY, 2> res3 = sycl::ext::oneapi::experimental::NAME(                \
        vec<bfloat16, 2>(arg[0], arg[0]), arg2.template swizzle<0, 0>());      \
    vec<RETTY, 2> res4 = sycl::ext::oneapi::experimental::NAME(                \
        arg.template swizzle<0, 0>(), vec<bfloat16, 2>(arg2[0], arg2[0]));     \
    /* Check the result. */                                                    \
    if (res2[0] != res[0] || res2[1] != res[0] || res3[0] != res[0] ||         \
        res3[1] != res[0] || res4[0] != res[0] || res4[1] != res[0]) {         \
      ERR[0] += 1;                                                             \
    }                                                                          \
    for (int i = 0; i < SZ; i++) {                                             \
      if (check(res[i], sycl::NAME(INPVAL, inpVal2))) {                        \
        ERR[0] += 1;                                                           \
      }                                                                        \
    }                                                                          \
  }

#define TEST_BUILTIN_VEC(NAME, SZ, RETTY, INPVAL, OPTEST)                      \
  { /* On Device */                                                            \
    buffer<int> err_buf(&err, 1);                                              \
    q.submit([&](handler &cgh) {                                               \
       accessor<int, 1, access::mode::write, target::device> ERR(err_buf,      \
                                                                 cgh);         \
       cgh.single_task([=]() { OPTEST(NAME, SZ, RETTY, INPVAL) });             \
     }).wait();                                                                \
  }                                                                            \
  assert(err == 0);                                                            \
  { /* On Host */                                                              \
    int ERR[1] = {0};                                                          \
    OPTEST(NAME, SZ, RETTY, INPVAL)                                            \
    assert(ERR[0] == 0);                                                       \
  }

#define TEST_BUILTIN_UNARY(NAME, RETTY, INPVAL)                                \
  TEST_BUILTIN_VEC(NAME, 1, RETTY, INPVAL, TEST_UNARY_OP)                      \
  TEST_BUILTIN_VEC(NAME, 2, RETTY, INPVAL, TEST_UNARY_OP)                      \
  TEST_BUILTIN_VEC(NAME, 3, RETTY, INPVAL, TEST_UNARY_OP)                      \
  TEST_BUILTIN_VEC(NAME, 4, RETTY, INPVAL, TEST_UNARY_OP)                      \
  TEST_BUILTIN_VEC(NAME, 8, RETTY, INPVAL, TEST_UNARY_OP)                      \
  TEST_BUILTIN_VEC(NAME, 16, RETTY, INPVAL, TEST_UNARY_OP)

#define TEST_BUILTIN_BINARY(NAME, RETTY, INPVAL)                               \
  TEST_BUILTIN_VEC(NAME, 1, RETTY, INPVAL, TEST_BINARY_OP)                     \
  TEST_BUILTIN_VEC(NAME, 2, RETTY, INPVAL, TEST_BINARY_OP)                     \
  TEST_BUILTIN_VEC(NAME, 3, RETTY, INPVAL, TEST_BINARY_OP)                     \
  TEST_BUILTIN_VEC(NAME, 4, RETTY, INPVAL, TEST_BINARY_OP)                     \
  TEST_BUILTIN_VEC(NAME, 8, RETTY, INPVAL, TEST_BINARY_OP)                     \
  TEST_BUILTIN_VEC(NAME, 16, RETTY, INPVAL, TEST_BINARY_OP)

void test() {
  queue q;
  int err = 0;
  float nan = std::nanf("");

  // Test isnan on host
  {
    vec<bfloat16, 3> arg{1.0f, nan, 2.0f};
    vec<int16_t, 3> res = sycl::ext::oneapi::experimental::isnan(arg);
    assert((res[0] == 0 && res[1] == -1 && res[2] == 0) &&
           "isnan() failed on host for vec");

    // Test for swizzles
    vec<int16_t, 2> res2 = sycl::ext::oneapi::experimental::isnan(arg.lo());
    assert((res2[0] == 0 && res2[1] == -1) &&
           "isnan() failed on host for vec swizzles");
  }

  // Tets isnan on device.
  {
    buffer<int> err_buf(&err, 1);
    q.submit([&](handler &cgh) {
       accessor<int, 1, access::mode::write, target::device> ERR(err_buf, cgh);
       cgh.single_task([=]() {
         vec<bfloat16, 3> arg{1.0f, nan, 2.0f};
         vec<int16_t, 3> res = sycl::ext::oneapi::experimental::isnan(arg);
         if (res[0] != 0 || res[1] != -1 || res[2] != 0) {
           ERR[0] += 1;
         }
       });
     }).wait();
    assert(err == 0 && "isnan failed on device for vec");
  }

  // Unary math builtins.
  TEST_BUILTIN_UNARY(fabs, bfloat16, -1.0f);
  TEST_BUILTIN_UNARY(fabs, bfloat16, 1.0f);

  TEST_BUILTIN_UNARY(cos, bfloat16, 0.1f);
  TEST_BUILTIN_UNARY(sin, bfloat16, 0.2f);

  TEST_BUILTIN_UNARY(ceil, bfloat16, 0.9f);
  TEST_BUILTIN_UNARY(floor, bfloat16, 0.9f);
  TEST_BUILTIN_UNARY(trunc, bfloat16, 0.9f);
  TEST_BUILTIN_UNARY(exp, bfloat16, 0.9f);
  TEST_BUILTIN_UNARY(exp10, bfloat16, 0.9f);
  TEST_BUILTIN_UNARY(exp2, bfloat16, 0.9f);
  TEST_BUILTIN_UNARY(rint, bfloat16, 0.9f);

  TEST_BUILTIN_UNARY(sqrt, bfloat16, 0.9f);
  TEST_BUILTIN_UNARY(rsqrt, bfloat16, 0.9f);
  TEST_BUILTIN_UNARY(log, bfloat16, 20.0f);
  TEST_BUILTIN_UNARY(log2, bfloat16, 2.0f);
  TEST_BUILTIN_UNARY(log10, bfloat16, 2.0f);

  TEST_BUILTIN_BINARY(fmin, bfloat16, 0.9f);
  TEST_BUILTIN_BINARY(fmax, bfloat16, 0.9f);
  TEST_BUILTIN_BINARY(fmin, bfloat16, nan);
  TEST_BUILTIN_BINARY(fmax, bfloat16, nan);

  // Test fma operation on host.
  {
    vec<bfloat16, 3> arg1, arg2, arg3;
    bfloat16 inpVal1 = 1.0f;
    bfloat16 inpVal2 = 2.0f;
    bfloat16 inpVal3 = 3.0f;
    /* Initialize the vector with INPVAL */
    for (int i = 0; i < 3; i++) {
      arg1[i] = inpVal1;
      arg2[i] = inpVal2;
      arg3[i] = inpVal3;
    }
    /* Perform the operation. */
    auto res = sycl::ext::oneapi::experimental::fma(arg1, arg2, arg3);

    // Test different combination of vec an swizzle.
    auto res1 = sycl::ext::oneapi::experimental::fma(
        arg1.template swizzle<0, 0>(), arg2.template swizzle<0, 0>(),
        arg3.template swizzle<0, 0>());

    auto res2 = sycl::ext::oneapi::experimental::fma(
        vec<bfloat16, 2>(arg1[0], arg1[0]), arg2.template swizzle<0, 0>(),
        arg3.template swizzle<0, 0>());

    auto res3 = sycl::ext::oneapi::experimental::fma(
        arg1.template swizzle<0, 0>(), vec<bfloat16, 2>(arg2[0], arg2[0]),
        arg3.template swizzle<0, 0>());

    auto res4 = sycl::ext::oneapi::experimental::fma(
        arg1.template swizzle<0, 0>(), arg2.template swizzle<0, 0>(),
        vec<bfloat16, 2>(arg3[0], arg3[0]));

    /* Check the result. */
    if (res1[0] != res[0] || res1[1] != res[0] || res2[0] != res[0] ||
        res2[1] != res[0] || res3[0] != res[0] || res3[1] != res[0] ||
        res4[0] != res[0] || res4[1] != res[0]) {
      err += 1;
    }
    for (int i = 0; i < 3; i++) {
      if (check(res[i], sycl::ext::oneapi::experimental::fma(inpVal1, inpVal2,
                                                             inpVal3))) {
        err += 1;
      }
    }
    assert(err == 0);
  }

  // Test fma on device.
  {
    buffer<int> err_buf(&err, 1);
    q.submit([&](handler &cgh) {
       accessor<int, 1, access::mode::write, target::device> ERR(err_buf, cgh);
       cgh.single_task([=]() {
         vec<bfloat16, 3> arg1, arg2, arg3;
         bfloat16 inpVal1 = 1.0f;
         bfloat16 inpVal2 = 2.0f;
         bfloat16 inpVal3 = 3.0f;
         /* Initialize the vector with INPVAL */
         for (int i = 0; i < 3; i++) {
           arg1[i] = inpVal1;
           arg2[i] = inpVal2;
           arg3[i] = inpVal3;
         }
         /* Perform the operation. */
         auto res = sycl::ext::oneapi::experimental::fma(arg1, arg2, arg3);

         // Test different combination of vec an swizzle.
         auto res1 = sycl::ext::oneapi::experimental::fma(
             arg1.template swizzle<0, 0>(), arg2.template swizzle<0, 0>(),
             arg3.template swizzle<0, 0>());

         auto res2 = sycl::ext::oneapi::experimental::fma(
             vec<bfloat16, 2>(arg1[0], arg1[0]), arg2.template swizzle<0, 0>(),
             arg3.template swizzle<0, 0>());

         auto res3 = sycl::ext::oneapi::experimental::fma(
             arg1.template swizzle<0, 0>(), vec<bfloat16, 2>(arg2[0], arg2[0]),
             arg3.template swizzle<0, 0>());

         auto res4 = sycl::ext::oneapi::experimental::fma(
             arg1.template swizzle<0, 0>(), arg2.template swizzle<0, 0>(),
             vec<bfloat16, 2>(arg3[0], arg3[0]));

         /* Check the result. */
         if (res1[0] != res[0] || res1[1] != res[0] || res2[0] != res[0] ||
             res2[1] != res[0] || res3[0] != res[0] || res3[1] != res[0] ||
             res4[0] != res[0] || res4[1] != res[0]) {
           ERR[0] += 1;
         }
         for (int i = 0; i < 3; i++) {
           if (check(res[i], sycl::ext::oneapi::experimental::fma(
                                 inpVal1, inpVal2, inpVal3))) {
             ERR[0] += 1;
           }
         }
       });
     }).wait();
    assert(err == 0);
  }
}

int main() {

  test();
  return 0;
}
