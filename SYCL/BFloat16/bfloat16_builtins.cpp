// REQUIRES: cuda
//
// Currently this test fails to compile for backends other than cuda.
// Other backends could use this test when bfloat16 math function support is
// added.
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out -Xsycl-target-backend --cuda-gpu-arch=sm_80
// RUN: %t.out
#include <sycl/ext/oneapi/experimental/builtins.hpp>
#include <sycl/sycl.hpp>

#include <cmath>
#include <vector>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

constexpr int N = 60; // divisible by all tested array sizes
constexpr float bf16_eps = 0.00390625;

float make_fp32(uint16_t x) {
  uint32_t y = x;
  y = y << 16;
  auto res = reinterpret_cast<float *>(&y);
  return *res;
}

bool check(float a, float b) {
  return fabs(2 * (a - b) / (a + b)) > bf16_eps * 2;
}

#define TEST_BUILTIN_1_SCAL_IMPL(NAME)                                         \
  {                                                                            \
    buffer<float> a_buf(&a[0], N);                                             \
    buffer<int> err_buf(&err, 1);                                              \
    q.submit([&](handler &cgh) {                                               \
      accessor<float, 1, access::mode::read_write, target::device> A(a_buf,    \
                                                                     cgh);     \
      accessor<int, 1, access::mode::write, target::device> ERR(err_buf, cgh); \
      cgh.parallel_for(N, [=](id<1> index) {                                   \
        if (check(NAME(bfloat16{A[index]}), NAME(A[index]))) {                 \
          ERR[0] = 1;                                                          \
        }                                                                      \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  assert(err == 0);

#define TEST_BUILTIN_1_ARR_IMPL(NAME, SZ)                                      \
  {                                                                            \
    buffer<float, 2> a_buf{range<2>{N / SZ, SZ}};                              \
    buffer<int> err_buf(&err, 1);                                              \
    q.submit([&](handler &cgh) {                                               \
      accessor<float, 2, access::mode::read_write, target::device> A(a_buf,    \
                                                                     cgh);     \
      accessor<int, 1, access::mode::write, target::device> ERR(err_buf, cgh); \
      cgh.parallel_for(N / SZ, [=](id<1> index) {                              \
        marray<bfloat16, SZ> arg;                                              \
        for (int i = 0; i < SZ; i++) {                                         \
          arg[i] = A[index][i];                                                \
        }                                                                      \
        marray<bfloat16, SZ> res = NAME(arg);                                  \
        for (int i = 0; i < SZ; i++) {                                         \
          if (check(res[i], NAME(A[index][i]))) {                              \
            ERR[0] = 1;                                                        \
          }                                                                    \
        }                                                                      \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  assert(err == 0);

#define TEST_BUILTIN_1(NAME)                                                   \
  TEST_BUILTIN_1_SCAL_IMPL(NAME)                                               \
  TEST_BUILTIN_1_ARR_IMPL(NAME, 1)                                             \
  TEST_BUILTIN_1_ARR_IMPL(NAME, 2)                                             \
  TEST_BUILTIN_1_ARR_IMPL(NAME, 3)                                             \
  TEST_BUILTIN_1_ARR_IMPL(NAME, 4)                                             \
  TEST_BUILTIN_1_ARR_IMPL(NAME, 5)

#define TEST_BUILTIN_2_SCAL_IMPL(NAME)                                         \
  {                                                                            \
    buffer<float> a_buf(&a[0], N);                                             \
    buffer<float> b_buf(&b[0], N);                                             \
    buffer<int> err_buf(&err, 1);                                              \
    q.submit([&](handler &cgh) {                                               \
      accessor<float, 1, access::mode::read_write, target::device> A(a_buf,    \
                                                                     cgh);     \
      accessor<float, 1, access::mode::read_write, target::device> B(b_buf,    \
                                                                     cgh);     \
      accessor<int, 1, access::mode::write, target::device> ERR(err_buf, cgh); \
      cgh.parallel_for(N, [=](id<1> index) {                                   \
        if (check(NAME(bfloat16{A[index]}, bfloat16{B[index]}),                \
                  NAME(A[index], B[index]))) {                                 \
          ERR[0] = 1;                                                          \
        }                                                                      \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  assert(err == 0);

#define TEST_BUILTIN_2_ARR_IMPL(NAME, SZ)                                      \
  {                                                                            \
    buffer<float, 2> a_buf{range<2>{N / SZ, SZ}};                              \
    buffer<float, 2> b_buf{range<2>{N / SZ, SZ}};                              \
    buffer<int> err_buf(&err, 1);                                              \
    q.submit([&](handler &cgh) {                                               \
      accessor<float, 2, access::mode::read_write, target::device> A(a_buf,    \
                                                                     cgh);     \
      accessor<float, 2, access::mode::read_write, target::device> B(b_buf,    \
                                                                     cgh);     \
      accessor<int, 1, access::mode::write, target::device> ERR(err_buf, cgh); \
      cgh.parallel_for(N / SZ, [=](id<1> index) {                              \
        marray<bfloat16, SZ> arg0, arg1;                                       \
        for (int i = 0; i < SZ; i++) {                                         \
          arg0[i] = A[index][i];                                               \
          arg1[i] = B[index][i];                                               \
        }                                                                      \
        marray<bfloat16, SZ> res = NAME(arg0, arg1);                           \
        for (int i = 0; i < SZ; i++) {                                         \
          if (check(res[i], NAME(A[index][i], B[index][i]))) {                 \
            ERR[0] = 1;                                                        \
          }                                                                    \
        }                                                                      \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  assert(err == 0);

#define TEST_BUILTIN_2(NAME)                                                   \
  TEST_BUILTIN_2_SCAL_IMPL(NAME)                                               \
  TEST_BUILTIN_2_ARR_IMPL(NAME, 1)                                             \
  TEST_BUILTIN_2_ARR_IMPL(NAME, 2)                                             \
  TEST_BUILTIN_2_ARR_IMPL(NAME, 3)                                             \
  TEST_BUILTIN_2_ARR_IMPL(NAME, 4)                                             \
  TEST_BUILTIN_2_ARR_IMPL(NAME, 5)

#define TEST_BUILTIN_3_SCAL_IMPL(NAME)                                         \
  {                                                                            \
    buffer<float> a_buf(&a[0], N);                                             \
    buffer<float> b_buf(&b[0], N);                                             \
    buffer<float> c_buf(&c[0], N);                                             \
    buffer<int> err_buf(&err, 1);                                              \
    q.submit([&](handler &cgh) {                                               \
      accessor<float, 1, access::mode::read_write, target::device> A(a_buf,    \
                                                                     cgh);     \
      accessor<float, 1, access::mode::read_write, target::device> B(b_buf,    \
                                                                     cgh);     \
      accessor<float, 1, access::mode::read_write, target::device> C(c_buf,    \
                                                                     cgh);     \
      accessor<int, 1, access::mode::write, target::device> ERR(err_buf, cgh); \
      cgh.parallel_for(N, [=](id<1> index) {                                   \
        if (check(NAME(bfloat16{A[index]}, bfloat16{B[index]},                 \
                       bfloat16{C[index]}),                                    \
                  NAME(A[index], B[index], C[index]))) {                       \
          ERR[0] = 1;                                                          \
        }                                                                      \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  assert(err == 0);

#define TEST_BUILTIN_3_ARR_IMPL(NAME, SZ)                                      \
  {                                                                            \
    buffer<float, 2> a_buf{range<2>{N / SZ, SZ}};                              \
    buffer<float, 2> b_buf{range<2>{N / SZ, SZ}};                              \
    buffer<float, 2> c_buf{range<2>{N / SZ, SZ}};                              \
    buffer<int> err_buf(&err, 1);                                              \
    q.submit([&](handler &cgh) {                                               \
      accessor<float, 2, access::mode::read_write, target::device> A(a_buf,    \
                                                                     cgh);     \
      accessor<float, 2, access::mode::read_write, target::device> B(b_buf,    \
                                                                     cgh);     \
      accessor<float, 2, access::mode::read_write, target::device> C(c_buf,    \
                                                                     cgh);     \
      accessor<int, 1, access::mode::write, target::device> ERR(err_buf, cgh); \
      cgh.parallel_for(N / SZ, [=](id<1> index) {                              \
        marray<bfloat16, SZ> arg0, arg1, arg2;                                 \
        for (int i = 0; i < SZ; i++) {                                         \
          arg0[i] = A[index][i];                                               \
          arg1[i] = B[index][i];                                               \
          arg2[i] = C[index][i];                                               \
        }                                                                      \
        marray<bfloat16, SZ> res = NAME(arg0, arg1, arg2);                     \
        for (int i = 0; i < SZ; i++) {                                         \
          if (check(res[i], NAME(A[index][i], B[index][i], C[index][i]))) {    \
            ERR[0] = 1;                                                        \
          }                                                                    \
        }                                                                      \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  assert(err == 0);

#define TEST_BUILTIN_3(NAME)                                                   \
  TEST_BUILTIN_3_SCAL_IMPL(NAME)                                               \
  TEST_BUILTIN_3_ARR_IMPL(NAME, 1)                                             \
  TEST_BUILTIN_3_ARR_IMPL(NAME, 2)                                             \
  TEST_BUILTIN_3_ARR_IMPL(NAME, 3)                                             \
  TEST_BUILTIN_3_ARR_IMPL(NAME, 4)                                             \
  TEST_BUILTIN_3_ARR_IMPL(NAME, 5)

#define TEST_BUILTIN_2_NAN(NAME)                                               \
  {                                                                            \
    buffer<int> err_buf(&err, 1);                                              \
    buffer<float> nan_buf(&check_nan, 1);                                      \
    q.submit([&](handler &cgh) {                                               \
      accessor<int, 1, access::mode::write, target::device> ERR(err_buf, cgh); \
      accessor<float, 1, access::mode::write, target::device> checkNAN(        \
          nan_buf, cgh);                                                       \
      cgh.single_task([=]() {                                                  \
        checkNAN[0] = NAME(bfloat16{NAN}, bfloat16{NAN});                      \
        if ((NAME(bfloat16{2}, bfloat16{NAN}) != 2) ||                         \
            (NAME(bfloat16{NAN}, bfloat16{2}) != 2)) {                         \
          ERR[0] = 1;                                                          \
        }                                                                      \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  assert(err == 0);                                                            \
  assert(std::isnan(check_nan));

int main() {
  queue q;

  if (q.get_device().has(aspect::ext_oneapi_bfloat16)) {
    std::vector<float> a(N), b(N), c(N);
    int err = 0;

    for (int i = 0; i < N; i++) {
      a[i] = (i - N / 2) / (float)N;
      b[i] = (N / 2 - i) / (float)N;
      c[i] = (float)(3 * i);
    }

    TEST_BUILTIN_1(fabs);
    TEST_BUILTIN_2(fmin);
    TEST_BUILTIN_2(fmax);
    TEST_BUILTIN_3(fma);

    float check_nan = 0;
    TEST_BUILTIN_2_NAN(fmin);
    TEST_BUILTIN_2_NAN(fmax);
  }
  return 0;
}
