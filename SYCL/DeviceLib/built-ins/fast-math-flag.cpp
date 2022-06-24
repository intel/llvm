// RUN: %clangxx -fsycl -ffast-math -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <cassert>

#define __TEST_FFMATH_BINARY(func)                                             \
  int test_ffmath_##func() {                                                   \
    sycl::float4 r[2];                                                         \
    sycl::float4 val[2] = {{1.0004f, 1e-4f, 1.4f, 14.0f},                      \
                           {1.0004f, 1e-4f, 1.4f, 14.0f}};                     \
    {                                                                          \
      sycl::buffer<sycl::float4, 1> output(&r[0], sycl::range<1>(2));          \
      sycl::buffer<sycl::float4, 1> input(&val[0], sycl::range<1>(2));         \
      sycl::queue q;                                                           \
      q.submit([&](sycl::handler &cgh) {                                       \
        auto AccO =                                                            \
            output.template get_access<sycl::access::mode::write>(cgh);        \
        auto AccI = input.template get_access<sycl::access::mode::read>(cgh);  \
        cgh.single_task([=]() {                                                \
          AccO[0] = sycl::func(AccI[0], AccI[1]);                              \
          AccO[1] = sycl::native::func(AccI[0], AccI[1]);                      \
        });                                                                    \
      });                                                                      \
    }                                                                          \
    return sycl::all(r[0] == r[1]);                                            \
  }

#define __TEST_FFMATH_UNARY(func)                                              \
  int test_ffmath_##func() {                                                   \
    sycl::float4 val = {1.0004f, 1e-4f, 1.4f, 14.0f};                          \
    sycl::float4 r[2];                                                         \
    {                                                                          \
      sycl::buffer<sycl::float4, 1> output(&r[0], sycl::range<1>(2));          \
      sycl::buffer<sycl::float4, 1> input(&val, sycl::range<1>(1));            \
      sycl::queue q;                                                           \
      q.submit([&](sycl::handler &cgh) {                                       \
        auto AccO =                                                            \
            output.template get_access<sycl::access::mode::write>(cgh);        \
        auto AccI = input.template get_access<sycl::access::mode::read>(cgh);  \
        cgh.single_task([=]() {                                                \
          AccO[0] = sycl::func(AccI[0]);                                       \
          AccO[1] = sycl::native::func(AccI[0]);                               \
        });                                                                    \
      });                                                                      \
    }                                                                          \
    return sycl::all(r[0] == r[1]);                                            \
  }

__TEST_FFMATH_UNARY(cos)
__TEST_FFMATH_UNARY(exp)
__TEST_FFMATH_UNARY(exp2)
__TEST_FFMATH_UNARY(exp10)
__TEST_FFMATH_UNARY(log)
__TEST_FFMATH_UNARY(log2)
__TEST_FFMATH_UNARY(log10)
__TEST_FFMATH_BINARY(powr)
__TEST_FFMATH_UNARY(rsqrt)
__TEST_FFMATH_UNARY(sin)
__TEST_FFMATH_UNARY(sqrt)
__TEST_FFMATH_UNARY(tan)

int main() {

  assert(test_ffmath_cos());
  assert(test_ffmath_exp());
  assert(test_ffmath_exp2());
  assert(test_ffmath_exp10());
  assert(test_ffmath_log());
  assert(test_ffmath_log2());
  assert(test_ffmath_log10());
  assert(test_ffmath_powr());
  assert(test_ffmath_rsqrt());
  assert(test_ffmath_sin());
  assert(test_ffmath_sqrt());
  assert(test_ffmath_tan());

  return 0;
}
