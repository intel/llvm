// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}

// RUN: %{build} -Xclang -fdenormal-fp-math-f32="preserve-sign,preserve-sign" -o %t.out %{mathflags}
// RUN: %{run} %t.out

#include <cassert>
#include <type_traits>

#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>

using namespace sycl;

constexpr float eps = 1e-6;

template <typename T> bool checkClose(T A, T B) { return ((A - B) / A) < eps; }

#define __TEST_FFMATH_UNARY(func)                                              \
  template <typename T> void test_ffmath_##func(queue &deviceQueue) {          \
    T input[4] = {1.0004f, 1e-4f, 1.4f, 14.0f};                                \
    T res[4] = {-1, -1, -1, -1};                                               \
    {                                                                          \
      buffer<T> input_buff{input, sycl::range{4}};                             \
      buffer<T> res_buff{res, sycl::range{4}};                                 \
      deviceQueue.submit([&](handler &cgh) {                                   \
        accessor res_acc{res_buff, cgh};                                       \
        accessor input_acc{input_buff, cgh};                                   \
        cgh.parallel_for(sycl::range{4}, [=](sycl::id<1> idx) {                \
          res_acc[idx] = sycl::func(input_acc[idx]);                           \
        });                                                                    \
      });                                                                      \
    }                                                                          \
    for (auto i = 0; i < 4; ++i)                                               \
      assert(checkClose(res[i], sycl::func(input[i])));                        \
  }

#define __TEST_FFMATH_BINARY(func)                                             \
  template <typename T> void test_ffmath_##func(queue &deviceQueue) {          \
    T input[2][4] = {{1.0004f, 1e-4f, 1.4f, 14.0f},                            \
                     {1.0004f, 1e-4f, 1.4f, 14.0f}};                           \
    T res[4] = {-1, -1, -1, -1};                                               \
    {                                                                          \
      buffer<T> input1_buff{input[0], sycl::range{4}};                         \
      buffer<T> input2_buff{input[1], sycl::range{4}};                         \
      buffer<T> res_buff{res, sycl::range{4}};                                 \
      deviceQueue.submit([&](handler &cgh) {                                   \
        accessor res_acc{res_buff, cgh};                                       \
        accessor input1_acc{input1_buff, cgh};                                 \
        accessor input2_acc{input2_buff, cgh};                                 \
        cgh.parallel_for(sycl::range{4}, [=](sycl::id<1> idx) {                \
          res_acc[idx] = sycl::func(input1_acc[idx], input2_acc[idx]);         \
        });                                                                    \
      });                                                                      \
    }                                                                          \
    for (auto i = 0; i < 4; ++i)                                               \
      assert(checkClose(res[i], sycl::func(input[0][i], input[1][i])));        \
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
  test_ffmath_cos<float>(q);
  test_ffmath_exp<float>(q);
  test_ffmath_exp2<float>(q);
  test_ffmath_exp10<float>(q);
  test_ffmath_log<float>(q);
  test_ffmath_log2<float>(q);
  test_ffmath_log10<float>(q);
  test_ffmath_powr<float>(q);
  test_ffmath_rsqrt<float>(q);
  test_ffmath_sin<float>(q);
  test_ffmath_sqrt<float>(q);
  test_ffmath_tan<float>(q);
  test_ffmath_powr<float>(q);

  return 0;
}
