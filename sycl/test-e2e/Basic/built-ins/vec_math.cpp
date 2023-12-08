// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}

// RUN: %{build} %{mathflags} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes %{mathflags} -o %t2.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out %}

#include <sycl/sycl.hpp>

#define TEST(FUNC, VEC_ELEM_TYPE, DIM, EXPECTED, DELTA, ...)                   \
  {                                                                            \
    {                                                                          \
      VEC_ELEM_TYPE result[DIM];                                               \
      {                                                                        \
        sycl::buffer<VEC_ELEM_TYPE> b(result, sycl::range{DIM});               \
        deviceQueue.submit([&](sycl::handler &cgh) {                           \
          sycl::accessor res_access{b, cgh};                                   \
          cgh.single_task([=]() {                                              \
            sycl::vec<VEC_ELEM_TYPE, DIM> res = FUNC(__VA_ARGS__);             \
            for (int i = 0; i < DIM; i++)                                      \
              res_access[i] = res[i];                                          \
          });                                                                  \
        });                                                                    \
      }                                                                        \
      for (int i = 0; i < DIM; i++)                                            \
        assert(abs(result[i] - EXPECTED[i]) <= DELTA);                         \
    }                                                                          \
  }

#define TEST2(FUNC, VEC_ELEM_TYPE, PTR_TYPE, DIM, EXPECTED_1, EXPECTED_2,      \
              DELTA, ...)                                                      \
  {                                                                            \
    {                                                                          \
      VEC_ELEM_TYPE result[DIM];                                               \
      sycl::vec<PTR_TYPE, DIM> result_ptr;                                     \
      {                                                                        \
        sycl::buffer<VEC_ELEM_TYPE> b(result, sycl::range{DIM});               \
        sycl::buffer<sycl::vec<PTR_TYPE, DIM>, 1> b_ptr(&result_ptr,           \
                                                        sycl::range<1>(1));    \
        deviceQueue.submit([&](sycl::handler &cgh) {                           \
          sycl::accessor res_access{b, cgh};                                   \
          sycl::accessor res_ptr_access{b_ptr, cgh};                           \
          cgh.single_task([=]() {                                              \
            sycl::multi_ptr<sycl::vec<PTR_TYPE, DIM>,                          \
                            sycl::access::address_space::global_space,         \
                            sycl::access::decorated::no>                       \
                ptr(res_ptr_access);                                           \
            sycl::vec<VEC_ELEM_TYPE, DIM> res = FUNC(__VA_ARGS__, ptr);        \
            for (int i = 0; i < DIM; i++)                                      \
              res_access[i] = res[i];                                          \
          });                                                                  \
        });                                                                    \
      }                                                                        \
      for (int i = 0; i < DIM; i++) {                                          \
        assert(std::abs(result[i] - EXPECTED_1[i]) <= DELTA);                  \
        assert(std::abs(result_ptr[i] - EXPECTED_2[i]) <= DELTA);              \
      }                                                                        \
    }                                                                          \
  }

#define TEST3(FUNC, VEC_ELEM_TYPE, DIM, ...)                                   \
  {                                                                            \
    {                                                                          \
      VEC_ELEM_TYPE result[DIM];                                               \
      {                                                                        \
        sycl::buffer<VEC_ELEM_TYPE> b(result, sycl::range{DIM});               \
        deviceQueue.submit([&](sycl::handler &cgh) {                           \
          sycl::accessor res_access{b, cgh};                                   \
          cgh.single_task([=]() {                                              \
            sycl::vec<VEC_ELEM_TYPE, DIM> res = FUNC(__VA_ARGS__);             \
            for (int i = 0; i < DIM; i++)                                      \
              res_access[i] = res[i];                                          \
          });                                                                  \
        });                                                                    \
      }                                                                        \
    }                                                                          \
  }

#define EXPECTED(TYPE, ...) ((TYPE[]){__VA_ARGS__})

int main() {
  sycl::queue deviceQueue;

  sycl::vec<float, 2> va1{1.0f, 2.0f};
  sycl::vec<float, 2> va2{3.0f, 2.0f};
  sycl::vec<float, 3> va3{180, 180, 180};
  sycl::vec<int, 3> va4{1, 1, 1};
  sycl::vec<float, 3> va5{180, -180, -180};
  sycl::vec<float, 3> va6{1.4f, 4.2f, 5.3f};
  sycl::vec<uint32_t, 3> va7{1, 2, 3};
  sycl::vec<uint64_t, 3> va8{1, 2, 3};
  sycl::vec<float, 3> va9{1.0f, 2.0f, 1.0f};
  sycl::vec<float, 3> va10{3.0f, 2.0f, 1.0f};
  sycl::vec<float, 2> va11{180, 180};
  sycl::vec<int, 2> va12{1, 1};
  sycl::vec<float, 2> va13{1.4f, 4.2f};

  TEST(sycl::fabs, float, 3, EXPECTED(float, 180, 180, 180), 0, va5);
  TEST(sycl::ilogb, int, 3, EXPECTED(int, 7, 7, 7), 0, va3);
  TEST(sycl::fmax, float, 2, EXPECTED(float, 3.0f, 2.0f), 0, va1, va2);
  TEST(sycl::fmin, float, 2, EXPECTED(float, 1.0f, 2.0f), 0, va1, 5.0f);
  TEST(sycl::ldexp, float, 3, EXPECTED(float, 360, 360, 360), 0, va3, va4);
  TEST(sycl::rootn, float, 3, EXPECTED(float, 180, 180, 180), 0.1, va3, va4);
  TEST2(sycl::fract, float, float, 3, EXPECTED(float, 0.4f, 0.2f, 0.3f),
        EXPECTED(float, 1, 4, 5), 0.0001, va6);
  TEST2(sycl::modf, float, float, 3, EXPECTED(float, 0.4f, 0.2f, 0.3f),
        EXPECTED(float, 1, 4, 5), 0.0001, va6);
  TEST2(sycl::sincos, float, float, 3,
        EXPECTED(float, 0.98545f, -0.871576f, -0.832267f),
        EXPECTED(float, 0.169967, -0.490261, 0.554375), 0.0001, va6);
  TEST2(sycl::frexp, float, int, 3, EXPECTED(float, 0.7f, 0.525f, 0.6625f),
        EXPECTED(int, 1, 3, 3), 0.0001, va6);
  TEST2(sycl::lgamma_r, float, int, 3,
        EXPECTED(float, -0.119613f, 2.04856f, 3.63964f), EXPECTED(int, 1, 1, 1),
        0.0001, va6);
  TEST2(sycl::remquo, float, int, 3, EXPECTED(float, 1.4f, 4.2f, 5.3f),
        EXPECTED(int, 0, 0, 0), 0.0001, va6, va3);
  TEST3(sycl::nan, float, 3, va7);
  if (deviceQueue.get_device().has(sycl::aspect::fp64)) {
    TEST3(sycl::nan, double, 3, va8);
  }
  TEST(sycl::half_precision::exp10, float, 2, EXPECTED(float, 10, 100), 0.1,
       va1);

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  TEST(sycl::fabs, float, 2, EXPECTED(float, 180, 180), 0, va5.swizzle<0, 1>());
  TEST(sycl::ilogb, int, 2, EXPECTED(int, 7, 7), 0, va3.swizzle<0, 1>());
  TEST(sycl::fmax, float, 2, EXPECTED(float, 3.0f, 2.0f), 0,
       va9.swizzle<0, 1>(), va2);
  TEST(sycl::fmax, float, 2, EXPECTED(float, 3.0f, 2.0f), 0, va1,
       va10.swizzle<0, 1>());
  TEST(sycl::fmax, float, 2, EXPECTED(float, 3.0f, 2.0f), 0,
       va9.swizzle<0, 1>(), va10.swizzle<0, 1>());
  TEST(sycl::fmax, float, 2, EXPECTED(float, 5.0f, 5.0f), 0,
       va9.swizzle<0, 1>(), 5.0f);
  TEST(sycl::fmin, float, 2, EXPECTED(float, 1.0f, 2.0f), 0,
       va9.swizzle<0, 1>(), va2);
  TEST(sycl::fmin, float, 2, EXPECTED(float, 1.0f, 2.0f), 0, va1,
       va10.swizzle<0, 1>());
  TEST(sycl::fmin, float, 2, EXPECTED(float, 1.0f, 2.0f), 0,
       va9.swizzle<0, 1>(), va10.swizzle<0, 1>());
  TEST(sycl::fmin, float, 2, EXPECTED(float, 1.0f, 2.0f), 0,
       va9.swizzle<0, 1>(), 5.0f);
  TEST(sycl::ldexp, float, 2, EXPECTED(float, 360, 360), 0, va3.swizzle<0, 1>(),
       va12);
  TEST(sycl::ldexp, float, 2, EXPECTED(float, 360, 360), 0, va11,
       va4.swizzle<0, 1>());
  TEST(sycl::ldexp, float, 2, EXPECTED(float, 360, 360), 0, va3.swizzle<0, 1>(),
       va4.swizzle<0, 1>());
  TEST(sycl::ldexp, float, 3, EXPECTED(float, 5760, 5760, 5760), 0, va3, 5);
  TEST(sycl::ldexp, float, 2, EXPECTED(float, 5760, 5760), 0,
       va3.swizzle<0, 1>(), 5);
  TEST(sycl::pown, float, 3, EXPECTED(float, 180, 180, 180), 0.1, va3, va4);
  TEST(sycl::pown, float, 2, EXPECTED(float, 180, 180), 0.1,
       va3.swizzle<0, 1>(), va12);
  TEST(sycl::pown, float, 2, EXPECTED(float, 180, 180), 0.1, va11,
       va4.swizzle<0, 1>());
  TEST(sycl::pown, float, 2, EXPECTED(float, 180, 180), 0.1,
       va3.swizzle<0, 1>(), va4.swizzle<0, 1>());
  TEST(sycl::rootn, float, 3, EXPECTED(float, 180, 180, 180), 0.1, va3, va4);
  TEST(sycl::rootn, float, 2, EXPECTED(float, 180, 180), 0.1,
       va3.swizzle<0, 1>(), va12);
  TEST(sycl::rootn, float, 2, EXPECTED(float, 180, 180), 0.1, va11,
       va4.swizzle<0, 1>());
  TEST(sycl::rootn, float, 2, EXPECTED(float, 180, 180), 0.1,
       va3.swizzle<0, 1>(), va4.swizzle<0, 1>());
  TEST2(sycl::fract, float, float, 2, EXPECTED(float, 0.4f, 0.2f),
        EXPECTED(float, 1, 4), 0.0001, va6.swizzle<0, 1>());
  TEST2(sycl::modf, float, float, 2, EXPECTED(float, 0.4f, 0.2f),
        EXPECTED(float, 1, 4), 0.0001, va6.swizzle<0, 1>());
  TEST2(sycl::sincos, float, float, 2, EXPECTED(float, 0.98545f, -0.871576f),
        EXPECTED(float, 0.169967, -0.490261), 0.0001, va6.swizzle<0, 1>());
  TEST2(sycl::frexp, float, int, 2, EXPECTED(float, 0.7f, 0.525f),
        EXPECTED(int, 1, 3), 0.0001, va6.swizzle<0, 1>());
  TEST2(sycl::lgamma_r, float, int, 2, EXPECTED(float, -0.119613f, 2.04856f),
        EXPECTED(int, 1, 1), 0.0001, va6.swizzle<0, 1>());
  TEST2(sycl::remquo, float, int, 2, EXPECTED(float, 1.4f, 4.2f),
        EXPECTED(int, 0, 0), 0.0001, va6.swizzle<0, 1>(), va11);
  TEST2(sycl::remquo, float, int, 2, EXPECTED(float, 1.4f, 4.2f),
        EXPECTED(int, 0, 0), 0.0001, va13, va3.swizzle<0, 1>());
  TEST2(sycl::remquo, float, int, 2, EXPECTED(float, 1.4f, 4.2f),
        EXPECTED(int, 0, 0), 0.0001, va6.swizzle<0, 1>(), va3.swizzle<0, 1>());
  TEST3(sycl::nan, float, 2, va7.swizzle<0, 1>());
  if (deviceQueue.get_device().has(sycl::aspect::fp64)) {
    TEST3(sycl::nan, double, 2, va8.swizzle<0, 1>());
  }
  TEST(sycl::half_precision::exp10, float, 2, EXPECTED(float, 10, 100), 0.1,
       va9.swizzle<0, 1>());
#endif // __INTEL_PREVIEW_BREAKING_CHANGES

  return 0;
}
