
// On CUDA, the test behaves differently depending on whether it is compiled for
// sm_xx>=sm_80 or not:
// + sm_80 and above uses some native bfloat16 math instructions
// + below sm_80 always uses generic impls

// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}
// REQUIRES: aspect-ext_oneapi_bfloat16_math_functions
// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %if any-device-is-cuda %{ -Xsycl-target-backend --cuda-gpu-arch=sm_80 %} %s -o %t.out %{mathflags}
// RUN: %{run} %t.out

// Test "new" (ABI breaking) for all platforms ( sm_80/native if CUDA )
// RUN:  %if preview-breaking-changes-supported %{  %clangxx -fsycl -fpreview-breaking-changes -fsycl-targets=%{sycl_triple} %if any-device-is-cuda %{ -Xsycl-target-backend --cuda-gpu-arch=sm_80 %} %s -o %t2.out %{mathflags} %}
// RUN:  %if preview-breaking-changes-supported %{  %{run} %t2.out  %}

// If CUDA, test "new" again for sm_75/generic
// RUN:  %if any-device-is-cuda %{ %if preview-breaking-changes-supported %{  %clangxx -fsycl -fpreview-breaking-changes -fsycl-targets=%{sycl_triple}  -Xsycl-target-backend --cuda-gpu-arch=sm_75  %s -o %t3.out %{mathflags} %} %}
// RUN:  %if any-device-is-cuda %{ %if preview-breaking-changes-supported %{  %{run} %t3.out  %} %}

// Currently the feature isn't supported on FPGA.
// UNSUPPORTED: accelerator
#include <sycl/sycl.hpp>

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
  auto res = reinterpret_cast<float *>(&y);
  return *res;
}

bool check(float a, float b) {
  return fabs(2 * (a - b) / (a + b)) > bf16_eps * 2;
}

bool check(bool a, bool b) { return (a != b); }

#define TEST_BUILTIN_1_SCAL_IMPL(NAME)                                         \
  {                                                                            \
    buffer<float> a_buf(&a[0], N);                                             \
    buffer<int> err_buf(&err, 1);                                              \
    q.submit([&](handler &cgh) {                                               \
      accessor<float, 1, access::mode::read_write, target::device> A(a_buf,    \
                                                                     cgh);     \
      accessor<int, 1, access::mode::write, target::device> ERR(err_buf, cgh); \
      cgh.parallel_for(N, [=](id<1> index) {                                   \
        float ABF16 = float{bfloat16{A[index]}};                               \
        if (check(sycl::ext::oneapi::experimental::NAME(bfloat16{A[index]}),   \
                  sycl::NAME(ABF16))) {                                        \
          ERR[0] = 1;                                                          \
        }                                                                      \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  assert(err == 0);

#define TEST_BUILTIN_1_ARR_IMPL(NAME, SZ, RETTY)                               \
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
        marray<RETTY, SZ> res = NAME(arg);                                     \
        for (int i = 0; i < SZ; i++) {                                         \
          float ABF16 = float{bfloat16{A[index][i]}};                          \
          if (check(res[i], sycl::NAME(ABF16))) {                              \
            ERR[0] = 1;                                                        \
          }                                                                    \
        }                                                                      \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  assert(err == 0);

#define TEST_BUILTIN_1(NAME, RETTY)                                            \
  TEST_BUILTIN_1_SCAL_IMPL(NAME)                                               \
  TEST_BUILTIN_1_ARR_IMPL(NAME, 1, RETTY)                                      \
  TEST_BUILTIN_1_ARR_IMPL(NAME, 2, RETTY)                                      \
  TEST_BUILTIN_1_ARR_IMPL(NAME, 3, RETTY)                                      \
  TEST_BUILTIN_1_ARR_IMPL(NAME, 4, RETTY)                                      \
  TEST_BUILTIN_1_ARR_IMPL(NAME, 5, RETTY)

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
        float ABF16 = float{bfloat16{A[index]}};                               \
        float BBF16 = float{bfloat16{B[index]}};                               \
        if (check(NAME(bfloat16{A[index]}, bfloat16{B[index]}),                \
                  NAME(ABF16, BBF16))) {                                       \
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
          float ABF16 = float{bfloat16{A[index][i]}};                          \
          float BBF16 = float{bfloat16{B[index][i]}};                          \
          if (check(res[i], NAME(ABF16, BBF16))) {                             \
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
        float ABF16 = float{bfloat16{A[index]}};                               \
        float BBF16 = float{bfloat16{B[index]}};                               \
        float CBF16 = float{bfloat16{C[index]}};                               \
        if (check(NAME(bfloat16{A[index]}, bfloat16{B[index]},                 \
                       bfloat16{C[index]}),                                    \
                  NAME(ABF16, BBF16, CBF16))) {                                \
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
          float ABF16 = float{bfloat16{A[index][i]}};                          \
          float BBF16 = float{bfloat16{B[index][i]}};                          \
          float CBF16 = float{bfloat16{C[index][i]}};                          \
          if (check(res[i], NAME(ABF16, BBF16, CBF16))) {                      \
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

  std::vector<float> a(N), b(N), c(N);
  int err = 0;

  for (int i = 0; i < N; i++) {
    a[i] = (i - N / 2) / (float)N;
    b[i] = (N / 2 - i) / (float)N;
    c[i] = (float)(3 * i);
  }

  TEST_BUILTIN_1(fabs, bfloat16);
  TEST_BUILTIN_2(fmin);
  TEST_BUILTIN_2(fmax);
  TEST_BUILTIN_3(fma);

  float check_nan = 0;
  TEST_BUILTIN_2_NAN(fmin);
  TEST_BUILTIN_2_NAN(fmax);

  // Insert NAN value in a to test isnan
  a[0] = a[N - 1] = NAN;
  TEST_BUILTIN_1(isnan, bool);

  // Orignal input 'a[0...N-1]' are in range [-0.5, 0.5),
  // need to update it for generic math testing.
  // sin, cos testing
  for (int i = 0; i < N; ++i) {
    a[i] = (i / (float)(N - 1)) * 6.28;
    if ((i & 0x1) == 0x1)
      a[i] = -a[i];
  }
  TEST_BUILTIN_1(cos, sycl::ext::oneapi::bfloat16);
  TEST_BUILTIN_1(sin, sycl::ext::oneapi::bfloat16);

  // ceil, floor, trunc, exp, exp2, exp10, rint testing
  TEST_BUILTIN_1(ceil, sycl::ext::oneapi::bfloat16);
  TEST_BUILTIN_1(floor, sycl::ext::oneapi::bfloat16);
  TEST_BUILTIN_1(trunc, sycl::ext::oneapi::bfloat16);
  TEST_BUILTIN_1(exp, sycl::ext::oneapi::bfloat16);
  TEST_BUILTIN_1(exp10, sycl::ext::oneapi::bfloat16);
  TEST_BUILTIN_1(exp2, sycl::ext::oneapi::bfloat16);
  TEST_BUILTIN_1(rint, sycl::ext::oneapi::bfloat16);

  // log, log2, log10, sqrt, rsqrt testing, the input
  // must be positive.
  for (int i = 0; i < N; ++i)
    a[i] = a[i] + 8.5;
  TEST_BUILTIN_1(sqrt, sycl::ext::oneapi::bfloat16);
  TEST_BUILTIN_1(rsqrt, sycl::ext::oneapi::bfloat16);
  TEST_BUILTIN_1(log, sycl::ext::oneapi::bfloat16);
  TEST_BUILTIN_1(log2, sycl::ext::oneapi::bfloat16);
  TEST_BUILTIN_1(log10, sycl::ext::oneapi::bfloat16);

  return 0;
}
