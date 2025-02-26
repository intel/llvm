// REQUIRES: aspect-fp16
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>

#include <cmath>
#include <limits>

using namespace sycl;

constexpr int SZ_max = 16;

bool check(float a, float b) {
  return sycl::fabs(2 * (a - b) / (a + b)) <
             std::numeric_limits<half>::epsilon() ||
         a < std::numeric_limits<half>::min();
}

template <int N> bool check(vec<float, N> a, vec<float, N> b) {
  for (int i = 0; i < N; i++) {
    if (!check(a[i], b[i])) {
      return false;
    }
  }
  return true;
}

#define TEST_BUILTIN_1_VEC_IMPL(NAME, SZ)                                      \
  {                                                                            \
    float##SZ *a = (float##SZ *)&A[0];                                         \
    float##SZ *b = (float##SZ *)&B[0];                                         \
    if (i < SZ_max / SZ) {                                                     \
      if (!check(NAME(a[i]), NAME(a[i].convert<half>()).convert<float>())) {   \
        err[0] = 1;                                                            \
      }                                                                        \
    }                                                                          \
  }

// vectors of size 3 need separate test, as they actually have the size of 4
// elements
#define TEST_BUILTIN_1_VEC3_IMPL(NAME)                                         \
  {                                                                            \
    float3 *a = (float3 *)&A[0];                                               \
    float3 *b = (float3 *)&B[0];                                               \
    if (i < SZ_max / 4) {                                                      \
      if (!check(NAME(a[i]), NAME(a[i].convert<half>()).convert<float>())) {   \
        err[0] = 1;                                                            \
      }                                                                        \
    }                                                                          \
  }

#define TEST_BUILTIN_1_SCAL_IMPL(NAME)                                         \
  {                                                                            \
    float *a = (float *)&A[0];                                                 \
    float *b = (float *)&B[0];                                                 \
    if (!check(NAME(a[i]), (float)NAME((half)a[i]))) {                         \
      err[0] = 1;                                                              \
    }                                                                          \
  }

#define TEST_BUILTIN_1(NAME)                                                   \
  TEST_BUILTIN_1_SCAL_IMPL(NAME)                                               \
  TEST_BUILTIN_1_VEC_IMPL(NAME, 2)                                             \
  TEST_BUILTIN_1_VEC3_IMPL(NAME)                                               \
  TEST_BUILTIN_1_VEC_IMPL(NAME, 4)                                             \
  TEST_BUILTIN_1_VEC_IMPL(NAME, 8)                                             \
  TEST_BUILTIN_1_VEC_IMPL(NAME, 16)

#define TEST_BUILTIN_2_VEC_IMPL(NAME, SZ)                                      \
  {                                                                            \
    float##SZ *a = (float##SZ *)&A[0];                                         \
    float##SZ *b = (float##SZ *)&B[0];                                         \
    if (i < SZ_max / SZ) {                                                     \
      if (!check(NAME(a[i], b[i]),                                             \
                 NAME(a[i].convert<half>(), b[i].convert<half>())              \
                     .convert<float>())) {                                     \
        err[0] = 1;                                                            \
      }                                                                        \
    }                                                                          \
  }

#define TEST_BUILTIN_2_VEC3_IMPL(NAME)                                         \
  {                                                                            \
    float3 *a = (float3 *)&A[0];                                               \
    float3 *b = (float3 *)&B[0];                                               \
    if (i < SZ_max / 4) {                                                      \
      if (!check(NAME(a[i], b[i]),                                             \
                 NAME(a[i].convert<half>(), b[i].convert<half>())              \
                     .convert<float>())) {                                     \
        err[0] = 1;                                                            \
      }                                                                        \
    }                                                                          \
  }

#define TEST_BUILTIN_2_SCAL_IMPL(NAME)                                         \
  {                                                                            \
    float *a = (float *)&A[0];                                                 \
    float *b = (float *)&B[0];                                                 \
    if (!check(NAME(a[i], b[i]), (float)NAME((half)a[i], (half)b[i]))) {       \
      err[0] = 1;                                                              \
    }                                                                          \
  }

#define TEST_BUILTIN_2(NAME)                                                   \
  TEST_BUILTIN_2_SCAL_IMPL(NAME)                                               \
  TEST_BUILTIN_2_VEC_IMPL(NAME, 2)                                             \
  TEST_BUILTIN_2_VEC3_IMPL(NAME)                                               \
  TEST_BUILTIN_2_VEC_IMPL(NAME, 4)                                             \
  TEST_BUILTIN_2_VEC_IMPL(NAME, 8)                                             \
  TEST_BUILTIN_2_VEC_IMPL(NAME, 16)

#define TEST_BUILTIN_3_VEC_IMPL(NAME, SZ)                                      \
  {                                                                            \
    float##SZ *a = (float##SZ *)&A[0];                                         \
    float##SZ *b = (float##SZ *)&B[0];                                         \
    float##SZ *c = (float##SZ *)&C[0];                                         \
    if (i < SZ_max / SZ) {                                                     \
      if (!check(NAME(a[i], b[i], c[i]),                                       \
                 NAME(a[i].convert<half>(), b[i].convert<half>(),              \
                      c[i].convert<half>())                                    \
                     .convert<float>())) {                                     \
        err[0] = 1;                                                            \
      }                                                                        \
    }                                                                          \
  }

#define TEST_BUILTIN_3_VEC3_IMPL(NAME)                                         \
  {                                                                            \
    float3 *a = (float3 *)&A[0];                                               \
    float3 *b = (float3 *)&B[0];                                               \
    float3 *c = (float3 *)&C[0];                                               \
    if (i < SZ_max / 4) {                                                      \
      if (!check(NAME(a[i], b[i], c[i]),                                       \
                 NAME(a[i].convert<half>(), b[i].convert<half>(),              \
                      c[i].convert<half>())                                    \
                     .convert<float>())) {                                     \
        err[0] = 1;                                                            \
      }                                                                        \
    }                                                                          \
  }

#define TEST_BUILTIN_3_SCAL_IMPL(NAME)                                         \
  {                                                                            \
    float *a = (float *)&A[0];                                                 \
    float *b = (float *)&B[0];                                                 \
    float *c = (float *)&C[0];                                                 \
    if (!check(NAME(a[i], b[i], c[i]),                                         \
               (float)NAME((half)a[i], (half)b[i], (half)c[i]))) {             \
      err[0] = 1;                                                              \
    }                                                                          \
  }

#define TEST_BUILTIN_3(NAME)                                                   \
  TEST_BUILTIN_3_SCAL_IMPL(NAME)                                               \
  TEST_BUILTIN_3_VEC_IMPL(NAME, 2)                                             \
  TEST_BUILTIN_3_VEC3_IMPL(NAME)                                               \
  TEST_BUILTIN_3_VEC_IMPL(NAME, 4)                                             \
  TEST_BUILTIN_3_VEC_IMPL(NAME, 8)                                             \
  TEST_BUILTIN_3_VEC_IMPL(NAME, 16)

int main() {
  queue q;

  float16 a, b, c, d;
  for (int i = 0; i < SZ_max; i++) {
    a[i] = i / (float)SZ_max;
    b[i] = (SZ_max - i) / (float)SZ_max;
    c[i] = (float)(3 * i);
  }
  int err = 0;
  {
    buffer<float16> a_buf(&a, 1);
    buffer<float16> b_buf(&b, 1);
    buffer<float16> c_buf(&c, 1);
    buffer<int> err_buf(&err, 1);
    q.submit([&](handler &cgh) {
      auto A = a_buf.get_access<access::mode::read>(cgh);
      auto B = b_buf.get_access<access::mode::read>(cgh);
      auto C = c_buf.get_access<access::mode::read>(cgh);
      auto err = err_buf.get_access<access::mode::write>(cgh);
      cgh.parallel_for(SZ_max, [=](item<1> index) {
        size_t i = index.get_id(0);
        TEST_BUILTIN_1(sycl::fabs);
        TEST_BUILTIN_2(sycl::fmin);
        TEST_BUILTIN_2(sycl::fmax);
        TEST_BUILTIN_3(sycl::fma);
      });
    });
  }
  assert(err == 0);

  return 0;
}
