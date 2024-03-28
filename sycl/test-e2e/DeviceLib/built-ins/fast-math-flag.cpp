// RUN: %{build} -ffast-math -o %t.out
// RUN: %{run} %t.out

#include <cassert>
#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>

using namespace sycl;

template <typename T> bool checkEqual(vec<T, 4> A, vec<T, 4> B) {

  return sycl::all(A == B);
}

template <typename T, size_t N>
bool checkEqual(marray<T, N> A, marray<T, N> B) {
  for (int i = 0; i < N; i++) {
    if (A[i] != B[i]) {
      return false;
    }
  }
  return true;
}

#define __TEST_FFMATH_UNARY(func)                                              \
  template <typename T> void test_ffmath_##func(queue &deviceQueue) {          \
    T input{1.0004f, 1e-4f, 1.4f, 14.0f};                                      \
    T res[2] = {{-1, -1, -1, -1}, {-2, -2, -2, -2}};                           \
    {                                                                          \
      buffer<T, 1> input_buff(&input, 1);                                      \
      buffer<T, 1> res_buff(&res[0], sycl::range<1>(2));                       \
      deviceQueue.submit([&](handler &cgh) {                                   \
        accessor<T, 1, access::mode::write, target::device> res_acc(res_buff,  \
                                                                    cgh);      \
        accessor<T, 1, access::mode::read, target::device> input_acc(          \
            input_buff, cgh);                                                  \
        cgh.single_task([=]() {                                                \
          res_acc[0] = sycl::native::func(input_acc[0]);                       \
          res_acc[1] = sycl::func(input_acc[0]);                               \
        });                                                                    \
      });                                                                      \
    }                                                                          \
    assert(checkEqual(res[0], res[1]));                                        \
  }

#define __TEST_FFMATH_BINARY(func)                                             \
  template <typename T> void test_ffmath_##func(queue &deviceQueue) {          \
    T input[2] = {{1.0004f, 1e-4f, 1.4f, 14.0f},                               \
                  {1.0004f, 1e-4f, 1.4f, 14.0f}};                              \
    T res[2] = {{-1, -1, -1, -1}, {-2, -2, -2, -2}};                           \
    {                                                                          \
      buffer<T, 1> input_buff(&input[0], range<1>(2));                         \
      buffer<T, 1> res_buff(&res[0], range<1>(2));                             \
      deviceQueue.submit([&](handler &cgh) {                                   \
        accessor<T, 1, access::mode::write, target::device> res_acc(res_buff,  \
                                                                    cgh);      \
        accessor<T, 1, access::mode::read, target::device> input_acc(          \
            input_buff, cgh);                                                  \
        cgh.single_task([=]() {                                                \
          res_acc[0] = sycl::native::func(input_acc[0], input_acc[1]);         \
          res_acc[1] = sycl::func(input_acc[0], input_acc[1]);                 \
        });                                                                    \
      });                                                                      \
    }                                                                          \
    assert(checkEqual(res[0], res[1]));                                        \
  }

__TEST_FFMATH_UNARY(cos)
__TEST_FFMATH_UNARY(exp)
__TEST_FFMATH_UNARY(exp2)
__TEST_FFMATH_UNARY(exp10)
__TEST_FFMATH_UNARY(log)
__TEST_FFMATH_UNARY(log2)
__TEST_FFMATH_UNARY(log10)
__TEST_FFMATH_UNARY(rsqrt)
__TEST_FFMATH_UNARY(sin)
__TEST_FFMATH_UNARY(sqrt)
__TEST_FFMATH_UNARY(tan)

__TEST_FFMATH_BINARY(powr)

int main() {

  queue q;
  test_ffmath_cos<marray<float, 4>>(q);
  test_ffmath_exp<marray<float, 4>>(q);
  test_ffmath_exp2<marray<float, 4>>(q);
  test_ffmath_exp10<marray<float, 4>>(q);
  test_ffmath_log<marray<float, 4>>(q);
  test_ffmath_log2<marray<float, 4>>(q);
  test_ffmath_log10<marray<float, 4>>(q);
  test_ffmath_powr<marray<float, 4>>(q);
  test_ffmath_rsqrt<marray<float, 4>>(q);
  test_ffmath_sin<marray<float, 4>>(q);
  test_ffmath_sqrt<marray<float, 4>>(q);
  test_ffmath_tan<marray<float, 4>>(q);
  test_ffmath_powr<marray<float, 4>>(q);

  test_ffmath_cos<float4>(q);
  test_ffmath_exp<float4>(q);
  test_ffmath_exp2<float4>(q);
  test_ffmath_exp10<float4>(q);
  test_ffmath_log<float4>(q);
  test_ffmath_log2<float4>(q);
  test_ffmath_log10<float4>(q);
  test_ffmath_powr<float4>(q);
  test_ffmath_rsqrt<float4>(q);
  test_ffmath_sin<float4>(q);
  test_ffmath_sqrt<float4>(q);
  test_ffmath_tan<float4>(q);
  test_ffmath_powr<float4>(q);

  return 0;
}
