// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

#define TEST(FUNC, TYPE, EXPECTED, N, ...)                                     \
  {                                                                            \
    {                                                                          \
      TYPE result[N];                                                          \
      {                                                                        \
        sycl::buffer<TYPE> b(result, sycl::range{N});                          \
        deviceQueue.submit([&](sycl::handler &cgh) {                           \
          sycl::accessor res_access{b, cgh};                                   \
          cgh.single_task([=]() {                                              \
            sycl::marray<TYPE, N> res = FUNC(__VA_ARGS__);                     \
            for (int i = 0; i < N; i++)                                        \
              res_access[i] = res[i];                                          \
          });                                                                  \
        });                                                                    \
      }                                                                        \
      for (int i = 0; i < N; i++)                                              \
        assert(result[i] == EXPECTED[i]);                                      \
    }                                                                          \
  }

#define EXPECTED(TYPE, ...) ((TYPE[]){__VA_ARGS__})

int main() {
  sycl::queue deviceQueue;

  sycl::marray<float, 2> ma1{1.0, 2.0};
  sycl::marray<float, 2> ma2{1.0, 2.0};
  sycl::marray<float, 2> ma3{2.0, 1.0};
  sycl::marray<float, 2> ma4{2.0, 2.0};
  sycl::marray<float, 3> ma5{2.0, 2.0, 1.0};
  sycl::marray<float, 3> ma6{1.0, 5.0, 8.0};
  sycl::marray<bool, 3> c(1, 0, 1);

  TEST(sycl::isequal, bool, EXPECTED(bool, 1, 1), 2, ma1, ma2);
  TEST(sycl::isnotequal, bool, EXPECTED(bool, 0, 0), 2, ma1, ma2);
  TEST(sycl::isgreater, bool, EXPECTED(bool, 0, 1), 2, ma1, ma3);
  TEST(sycl::isgreaterequal, bool, EXPECTED(bool, 0, 1), 2, ma1, ma4);
  TEST(sycl::isless, bool, EXPECTED(bool, 0, 1), 2, ma3, ma1);
  TEST(sycl::islessequal, bool, EXPECTED(bool, 0, 1), 2, ma4, ma1);
  TEST(sycl::islessgreater, bool, EXPECTED(bool, 0, 0), 2, ma1, ma2);
  TEST(sycl::isfinite, bool, EXPECTED(bool, 1, 1), 2, ma1);
  TEST(sycl::isinf, bool, EXPECTED(bool, 0, 0), 2, ma1);
  TEST(sycl::isnan, bool, EXPECTED(bool, 0, 0), 2, ma1);
  TEST(sycl::isnormal, bool, EXPECTED(bool, 1, 1), 2, ma1);
  TEST(sycl::isordered, bool, EXPECTED(bool, 1, 1), 2, ma1, ma2);
  TEST(sycl::isunordered, bool, EXPECTED(bool, 0, 0), 2, ma1, ma2);
  TEST(sycl::signbit, bool, EXPECTED(bool, 0, 0), 2, ma1);
  TEST(sycl::select, float, EXPECTED(float, 1.0, 2.0, 8.0), 3, ma5, ma6, c);

  return 0;
}
