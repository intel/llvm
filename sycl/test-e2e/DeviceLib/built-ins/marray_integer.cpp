// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>

#define TEST(FUNC, MARRAY_RET_TYPE, DIM, EXPECTED, ...)                        \
  {                                                                            \
    {                                                                          \
      MARRAY_RET_TYPE result[DIM];                                             \
      {                                                                        \
        sycl::buffer<MARRAY_RET_TYPE> b(result, sycl::range{DIM});             \
        deviceQueue.submit([&](sycl::handler &cgh) {                           \
          sycl::accessor res_access{b, cgh};                                   \
          cgh.single_task([=]() {                                              \
            sycl::marray<MARRAY_RET_TYPE, DIM> res = FUNC(__VA_ARGS__);        \
            for (int i = 0; i < DIM; i++)                                      \
              res_access[i] = res[i];                                          \
          });                                                                  \
        });                                                                    \
      }                                                                        \
      for (int i = 0; i < DIM; i++)                                            \
        assert(result[i] == EXPECTED[i]);                                      \
    }                                                                          \
  }

#define EXPECTED(TYPE, ...) ((TYPE[]){__VA_ARGS__})

int main() {
  sycl::queue deviceQueue;

  sycl::marray<int, 3> ma1{1, -5, 5};
  sycl::marray<char, 3> ma1_char{1, -5, 5};
  sycl::marray<signed char, 3> ma1_signed_char{1, -5, 5};
  sycl::marray<short, 3> ma1_short{1, -5, 5};
  sycl::marray<long int, 3> ma1_long_int{1, -5, 5};
  sycl::marray<long long int, 3> ma1_long_long_int{1, -5, 5};
  sycl::marray<int, 3> ma2{2, 3, 8};
  sycl::marray<int, 3> ma3{8, 0, 9};
  sycl::marray<unsigned int, 3> ma4{1, 5, 5};
  sycl::marray<unsigned char, 3> ma4_uchar{1, 5, 5};
  sycl::marray<unsigned short, 3> ma4_ushort{1, 5, 5};
  sycl::marray<unsigned long int, 3> ma4_ulong_int{1, 5, 5};
  sycl::marray<unsigned long long int, 3> ma4_ulong_long_int{1, 5, 5};
  sycl::marray<unsigned int, 3> ma5{2, 3, 8};
  sycl::marray<unsigned int, 3> ma6{8, 0, 9};
  sycl::marray<int8_t, 3> ma7{1, 5, 5};
  sycl::marray<uint8_t, 3> ma8{1, 5, 5};
  sycl::marray<uint8_t, 3> ma9{2, 3, 8};
  sycl::marray<int16_t, 3> ma10{1, 5, 5};
  sycl::marray<uint16_t, 3> ma11{1, 5, 5};
  sycl::marray<uint16_t, 3> ma12{2, 3, 8};
  sycl::marray<int32_t, 3> ma13{1, 5, 5};
  sycl::marray<int32_t, 3> ma14{2, 3, 8};
  sycl::marray<uint32_t, 3> ma15{1, 5, 5};
  sycl::marray<uint32_t, 3> ma16{2, 3, 8};

  // geninteger abs(geninteger x)
  TEST(sycl::abs, int, 3, EXPECTED(int, 1, 5, 5), ma1);
  TEST(sycl::abs, int, 3, EXPECTED(int, 1, 5, 5), ma4);
  TEST(sycl::abs, char, 3, EXPECTED(char, 1, 5, 5), ma1_char);
  TEST(sycl::abs, signed char, 3, EXPECTED(signed char, 1, 5, 5),
       ma1_signed_char);
  TEST(sycl::abs, unsigned char, 3, EXPECTED(unsigned char, 1, 5, 5),
       ma4_uchar);
  TEST(sycl::abs, short, 3, EXPECTED(short, 1, 5, 5), ma1_short);
  TEST(sycl::abs, unsigned short, 3, EXPECTED(unsigned short, 1, 5, 5),
       ma4_ushort);
  TEST(sycl::abs, long int, 3, EXPECTED(long int, 1, 5, 5), ma1_long_int);
  TEST(sycl::abs, unsigned long int, 3, EXPECTED(unsigned long int, 1, 5, 5),
       ma4_ulong_int);
  TEST(sycl::abs, long long int, 3, EXPECTED(long long int, 1, 5, 5),
       ma1_long_long_int);
  TEST(sycl::abs, unsigned long long int, 3,
       EXPECTED(unsigned long long int, 1, 5, 5), ma4_ulong_long_int);
  // ugeninteger abs_diff(geninteger x, geninteger y)
  TEST(sycl::abs_diff, int, 3, EXPECTED(int, 1, 8, 3), ma1, ma2);
  TEST(sycl::abs_diff, int, 3, EXPECTED(int, 1, 2, 3), ma4, ma5);
  // geninteger add_sat(geninteger x, geninteger y)
  TEST(sycl::add_sat, int, 3, EXPECTED(int, 3, -2, 13), ma1, ma2);
  TEST(sycl::add_sat, unsigned int, 3, EXPECTED(unsigned int, 3, 8, 13), ma4,
       ma5);
  // geninteger hadd(geninteger x, geninteger y)
  TEST(sycl::hadd, int, 3, EXPECTED(int, 1, -1, 6), ma1, ma2);
  TEST(sycl::hadd, unsigned int, 3, EXPECTED(unsigned int, 1, 4, 6), ma4, ma5);
  // geninteger rhadd(geninteger x, geninteger y)
  TEST(sycl::rhadd, int, 3, EXPECTED(int, 2, -1, 7), ma1, ma2);
  TEST(sycl::rhadd, unsigned int, 3, EXPECTED(unsigned int, 2, 4, 7), ma4, ma5);
  // geninteger clamp(geninteger x, geninteger minval, geninteger maxval)
  TEST(sycl::clamp, int, 3, EXPECTED(int, 2, 3, 8), ma1, ma2, ma2);
  TEST(sycl::clamp, unsigned int, 3, EXPECTED(unsigned int, 2, 3, 8), ma4, ma5,
       ma5);
  // geninteger clamp(geninteger x, sgeninteger minval, sgeninteger maxval)
  TEST(sycl::clamp, int, 3, EXPECTED(int, 4, 4, 4), ma1, 4, 4);
  TEST(sycl::clamp, unsigned int, 3, EXPECTED(unsigned int, 4, 4, 4), ma4,
       (unsigned int)4, (unsigned int)4);
  // geninteger clz(geninteger x)
  TEST(sycl::clz, int, 3, EXPECTED(int, 31, 0, 29), ma1);
  TEST(sycl::clz, unsigned int, 3, EXPECTED(unsigned int, 31, 29, 29), ma4);
  // geninteger ctz(geninteger x)
  TEST(sycl::ctz, int, 3, EXPECTED(int, 0, 0, 0), ma1);
  TEST(sycl::ctz, unsigned int, 3, EXPECTED(unsigned int, 0, 0, 0), ma4);
  // geninteger mad_hi(geninteger a, geninteger b, geninteger c)
  TEST(sycl::mad_hi, int, 3, EXPECTED(int, 8, -1, 9), ma1, ma2, ma3);
  TEST(sycl::mad_hi, unsigned int, 3, EXPECTED(unsigned int, 8, 0, 9), ma4, ma5,
       ma6);
  // geninteger mad_sat(geninteger a, geninteger b, geninteger c)
  TEST(sycl::mad_sat, int, 3, EXPECTED(int, 10, -15, 49), ma1, ma2, ma3);
  TEST(sycl::mad_sat, unsigned int, 3, EXPECTED(unsigned int, 10, 15, 49), ma4,
       ma5, ma6);
  // geninteger max(geninteger x, geninteger y)
  TEST(sycl::max, int, 3, EXPECTED(int, 2, 3, 8), ma1, ma2);
  TEST(sycl::max, unsigned int, 3, EXPECTED(unsigned int, 2, 5, 8), ma4, ma5);
  // geninteger max(geninteger x, sgeninteger y)
  TEST(sycl::max, int, 3, EXPECTED(int, 4, 4, 5), ma1, 4);
  TEST(sycl::max, unsigned int, 3, EXPECTED(unsigned int, 4, 5, 5), ma4,
       (unsigned int)4);
  // geninteger min(geninteger x, geninteger y)
  TEST(sycl::min, int, 3, EXPECTED(int, 1, -5, 5), ma1, ma2);
  TEST(sycl::min, unsigned int, 3, EXPECTED(unsigned int, 1, 3, 5), ma4, ma5);
  // geninteger min(geninteger x, sgeninteger y)
  TEST(sycl::min, int, 3, EXPECTED(int, 1, -5, 4), ma1, 4);
  TEST(sycl::min, unsigned int, 3, EXPECTED(unsigned int, 1, 4, 4), ma4,
       (unsigned int)4);
  // geninteger mul_hi(geninteger x, geninteger y)
  TEST(sycl::mul_hi, int, 3, EXPECTED(int, 0, -1, 0), ma1, ma2);
  TEST(sycl::mul_hi, unsigned int, 3, EXPECTED(unsigned int, 0, 0, 0), ma4,
       ma5);
  // geninteger rotate(geninteger v, geninteger i)
  TEST(sycl::rotate, int, 3, EXPECTED(int, 4, -33, 1280), ma1, ma2);
  TEST(sycl::rotate, unsigned int, 3, EXPECTED(unsigned int, 4, 40, 1280), ma4,
       ma5);
  // geninteger sub_sat(geninteger x, geninteger y)
  TEST(sycl::sub_sat, int, 3, EXPECTED(int, -1, -8, -3), ma1, ma2);
  TEST(sycl::sub_sat, unsigned int, 3, EXPECTED(unsigned int, 0, 2, 0), ma4,
       ma5);
  // ugeninteger16bit upsample(ugeninteger8bit hi, ugeninteger8bit lo)
  TEST(sycl::upsample, uint16_t, 3, EXPECTED(uint16_t, 258, 1283, 1288), ma8,
       ma9);
  // igeninteger16bit upsample(igeninteger8bit hi, ugeninteger8bit lo)
  TEST(sycl::upsample, int16_t, 3, EXPECTED(int16_t, 258, 1283, 1288), ma7,
       ma9);
  // ugeninteger32bit upsample(ugeninteger16bit hi, ugeninteger16bit lo)
  TEST(sycl::upsample, uint32_t, 3, EXPECTED(uint32_t, 65538, 327683, 327688),
       ma11, ma12);
  // igeninteger32bit upsample(igeninteger16bit hi, ugeninteger16bit lo)
  TEST(sycl::upsample, int32_t, 3, EXPECTED(int32_t, 65538, 327683, 327688),
       ma10, ma12);
  // ugeninteger64bit upsample(ugeninteger32bit hi, ugeninteger32bit lo)
  TEST(sycl::upsample, uint64_t, 3,
       EXPECTED(uint64_t, 4294967298, 21474836483, 21474836488), ma15, ma16);
  // igeninteger64bit upsample(igeninteger32bit hi, ugeninteger32bit lo)
  TEST(sycl::upsample, int64_t, 3,
       EXPECTED(int64_t, 4294967298, 21474836483, 21474836488), ma13, ma16);
  // geninteger popcount(geninteger x)
  TEST(sycl::popcount, int, 3, EXPECTED(int, 1, 31, 2), ma1);
  TEST(sycl::popcount, unsigned int, 3, EXPECTED(unsigned int, 1, 2, 2), ma4);
  // geninteger32bit mad24(geninteger32bit x, geninteger32bit y, geninteger32bit
  // z)
  TEST(sycl::mad24, int32_t, 3, EXPECTED(int32_t, 4, 18, 48), ma13, ma14, ma14);
  TEST(sycl::mad24, uint32_t, 3, EXPECTED(uint32_t, 4, 18, 48), ma15, ma16,
       ma16);
  // geninteger32bit mul24(geninteger32bit x, geninteger32bit y)
  TEST(sycl::mul24, int32_t, 3, EXPECTED(int32_t, 2, 15, 40), ma13, ma14);
  TEST(sycl::mul24, uint32_t, 3, EXPECTED(uint32_t, 2, 15, 40), ma15, ma16);

  return 0;
}
