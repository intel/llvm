#include <gtest/gtest.h>
#include <sycl/ext/oneapi/experimental/float_8bit/types.hpp>

#include <cmath>
#include <cstdint>
#include <limits>

using namespace sycl::ext::oneapi::experimental;
using sycl::ext::oneapi::bfloat16;

TEST(FP8E5M3ArrayCtor, HalfDefaultRounding) {
  const sycl::half vals[2] = {sycl::half(1.0f),
                              sycl::half(std::ldexp(1.0f, -14))};
  fp8_e5m3_x2 v(vals);

  EXPECT_EQ(sizeof(v.vals), 2u);
  EXPECT_EQ(v.vals[0], 0x78);
  EXPECT_EQ(v.vals[1], 0x08);
}

TEST(FP8E5M3ArrayCtor, HalfExplicitRounding) {
  const sycl::half vals[2] = {sycl::half(1.5f), sycl::half(0.75f)};
  fp8_e5m3_x2 v(vals, rounding::to_even);

  EXPECT_EQ(sizeof(v.vals), 2u);
  EXPECT_EQ(v.vals[0], 0x7C);
  EXPECT_EQ(v.vals[1], 0x74);
}

TEST(FP8E5M3ArrayCtor, Bfloat16DefaultRounding) {
  const bfloat16 vals[2] = {bfloat16(2.0f), bfloat16(std::ldexp(1.0f, -13))};
  fp8_e5m3_x2 v(vals);

  EXPECT_EQ(sizeof(v.vals), 2u);
  EXPECT_EQ(v.vals[0], 0x80);
  EXPECT_EQ(v.vals[1], 0x10);
}

TEST(FP8E5M3ArrayCtor, Bfloat16ExplicitRounding) {
  const bfloat16 vals[2] = {bfloat16(3.0f), bfloat16(0.5f)};
  fp8_e5m3_x2 v(vals, rounding::to_even);

  EXPECT_EQ(sizeof(v.vals), 2u);
  EXPECT_EQ(v.vals[0], 0x84);
  EXPECT_EQ(v.vals[1], 0x70);
}

TEST(FP8E5M3ArrayCtor, FloatDefaultRounding) {
  const float vals[2] = {114688.0f, std::ldexp(1.0f, -17)};
  fp8_e5m3_x2 v(vals);

  EXPECT_EQ(sizeof(v.vals), 2u);
  EXPECT_EQ(v.vals[0], 0xFE);
  EXPECT_EQ(v.vals[1], 0x01);
}

TEST(FP8E5M3ArrayCtor, FloatExplicitRounding) {
  const float vals[2] = {1.25f, 0.875f * std::ldexp(1.0f, -14)};
  fp8_e5m3_x2 v(vals, rounding::to_even);

  EXPECT_EQ(sizeof(v.vals), 2u);
  EXPECT_EQ(v.vals[0], 0x7A);
  EXPECT_EQ(v.vals[1], 0x07);
}

TEST(FP8E5M3ArrayCtor, DoubleDefaultRounding) {
  const double vals[2] = {8.0, 0.125};
  fp8_e5m3_x2 v(vals);

  EXPECT_EQ(sizeof(v.vals), 2u);
  EXPECT_EQ(v.vals[0], 0x90);
  EXPECT_EQ(v.vals[1], 0x60);
}

TEST(FP8E5M3MarrayCtor, HalfDefaultRounding) {
  const sycl::marray<sycl::half, 2> vals{sycl::half(1.75f), sycl::half(0.5f)};
  fp8_e5m3_x2 v(vals);

  EXPECT_EQ(sizeof(v.vals), 2u);
  EXPECT_EQ(v.vals[0], 0x7E);
  EXPECT_EQ(v.vals[1], 0x70);
}

TEST(FP8E5M3MarrayCtor, HalfExplicitRounding) {
  const sycl::marray<sycl::half, 2> vals{sycl::half(2.0f), sycl::half(0.125f)};
  fp8_e5m3_x2 v(vals, rounding::to_even);

  EXPECT_EQ(sizeof(v.vals), 2u);
  EXPECT_EQ(v.vals[0], 0x80);
  EXPECT_EQ(v.vals[1], 0x60);
}

TEST(FP8E5M3MarrayCtor, Bfloat16DefaultRounding) {
  const sycl::marray<bfloat16, 2> vals{bfloat16(3.0f), bfloat16(0.25f)};
  fp8_e5m3_x2 v(vals);

  EXPECT_EQ(sizeof(v.vals), 2u);
  EXPECT_EQ(v.vals[0], 0x84);
  EXPECT_EQ(v.vals[1], 0x68);
}

TEST(FP8E5M3MarrayCtor, Bfloat16ExplicitRounding) {
  const sycl::marray<bfloat16, 2> vals{bfloat16(6.0f), bfloat16(0.75f)};
  fp8_e5m3_x2 v(vals, rounding::to_even);

  EXPECT_EQ(sizeof(v.vals), 2u);
  EXPECT_EQ(v.vals[0], 0x8C);
  EXPECT_EQ(v.vals[1], 0x74);
}

TEST(FP8E5M3MarrayCtor, FloatDefaultRounding) {
  const sycl::marray<float, 2> vals{12.0f, 0.03125f};
  fp8_e5m3_x2 v(vals);

  EXPECT_EQ(sizeof(v.vals), 2u);
  EXPECT_EQ(v.vals[0], 0x94);
  EXPECT_EQ(v.vals[1], 0x50);
}

TEST(FP8E5M3MarrayCtor, FloatExplicitRounding) {
  const sycl::marray<float, 2> vals{1.25f, std::ldexp(0.375f, -14)};
  fp8_e5m3_x2 v(vals, rounding::to_even);

  EXPECT_EQ(sizeof(v.vals), 2u);
  EXPECT_EQ(v.vals[0], 0x7A);
  EXPECT_EQ(v.vals[1], 0x03);
}

TEST(FP8E5M3MarrayCtor, DoubleDefaultRounding) {
  const sycl::marray<double, 2> vals{16.0, 0.0625};
  fp8_e5m3_x2 v(vals);

  EXPECT_EQ(sizeof(v.vals), 2u);
  EXPECT_EQ(v.vals[0], 0x98);
  EXPECT_EQ(v.vals[1], 0x58);
}

TEST(FP8E5M3ScalarIntCtor, ShortValue) {
  const short val = 5;
  fp8_e5m3 v(val);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x8A);
}

TEST(FP8E5M3ScalarIntCtor, IntValue) {
  const int val = 7;
  fp8_e5m3 v(val);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x8E);
}

TEST(FP8E5M3ScalarIntCtor, LongValue) {
  const long val = 9;
  fp8_e5m3 v(val);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x91);
}

TEST(FP8E5M3ScalarIntCtor, LongLongValue) {
  const long long val = 10;
  fp8_e5m3 v(val);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x92);
}

TEST(FP8E5M3ScalarIntCtor, UnsignedShortValue) {
  const unsigned short val = 14;
  fp8_e5m3 v(val);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x96);
}

TEST(FP8E5M3ScalarIntCtor, UnsignedIntValue) {
  const unsigned int val = 15;
  fp8_e5m3 v(val);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x97);
}

TEST(FP8E5M3ScalarIntCtor, UnsignedLongValue) {
  const unsigned long val = 18;
  fp8_e5m3 v(val);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x99);
}

TEST(FP8E5M3ScalarIntCtor, UnsignedLongLongValue) {
  const unsigned long long val = 20;
  fp8_e5m3 v(val);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x9A);
}

TEST(FP8E5M3ScalarIntCtor, UnsignedLimitsSaturate) {
  const unsigned short usmax = std::numeric_limits<unsigned short>::max();
  const unsigned int uimax = std::numeric_limits<unsigned int>::max();
  const unsigned long ulmax = std::numeric_limits<unsigned long>::max();
  const unsigned long long ullmax =
      std::numeric_limits<unsigned long long>::max();

  fp8_e5m3 vus(usmax);
  fp8_e5m3 vui(uimax);
  fp8_e5m3 vul(ulmax);
  fp8_e5m3 vull(ullmax);

  EXPECT_EQ(sizeof(vus.vals), 1u);
  EXPECT_EQ(vus.vals[0], 0xFE);
  EXPECT_EQ(vui.vals[0], 0xFE);
  EXPECT_EQ(vul.vals[0], 0xFE);
  EXPECT_EQ(vull.vals[0], 0xFE);
}

TEST(FP8E5M3AssignOp, HalfValue) {
  fp8_e5m3 v(sycl::half(1.0f));
  v = sycl::half(1.125f);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x79);
}

TEST(FP8E5M3AssignOp, Bfloat16Value) {
  fp8_e5m3 v(bfloat16(1.0f));
  v = bfloat16(1.875f);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x7F);
}

TEST(FP8E5M3AssignOp, FloatValue) {
  fp8_e5m3 v(1.0f);
  v = 11.0f;

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x93);
}

TEST(FP8E5M3AssignOp, DoubleValue) {
  fp8_e5m3 v(1.0);
  v = 5.5;

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x8B);
}

TEST(FP8E5M3AssignOp, ShortValue) {
  fp8_e5m3 v(1);
  v = static_cast<short>(6);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x8C);
}

TEST(FP8E5M3AssignOp, IntValue) {
  fp8_e5m3 v(1);
  v = 12;

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x94);
}

TEST(FP8E5M3AssignOp, LongValue) {
  fp8_e5m3 v(1);
  v = static_cast<long>(13);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x95);
}

TEST(FP8E5M3AssignOp, LongLongValue) {
  fp8_e5m3 v(1);
  v = static_cast<long long>(17);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x98);
}

TEST(FP8E5M3AssignOp, UnsignedShortValue) {
  fp8_e5m3 v(1);
  v = static_cast<unsigned short>(21);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x9A);
}

TEST(FP8E5M3AssignOp, UnsignedIntValue) {
  fp8_e5m3 v(1);
  v = static_cast<unsigned int>(22);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x9B);
}

TEST(FP8E5M3AssignOp, UnsignedLongValue) {
  fp8_e5m3 v(1);
  v = static_cast<unsigned long>(24);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x9C);
}

TEST(FP8E5M3AssignOp, UnsignedLongLongValue) {
  fp8_e5m3 v(1);
  v = static_cast<unsigned long long>(26);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x9D);
}

TEST(FP8E5M3ConvertOp, HalfValue) {
  fp8_e5m3 v(2.5f);
  auto out = static_cast<sycl::half>(v);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x82);
  EXPECT_EQ(static_cast<float>(out), 2.5f);
}

TEST(FP8E5M3ConvertOp, Bfloat16Value) {
  fp8_e5m3 v(0.375f);
  auto out = static_cast<bfloat16>(v);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x6C);
  EXPECT_EQ(static_cast<float>(out), 0.375f);
}

TEST(FP8E5M3ConvertOp, FloatValue) {
  fp8_e5m3 v(2.25f);
  float out = static_cast<float>(v);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x81);
  EXPECT_EQ(out, 2.25f);
}

TEST(FP8E5M3ConvertOp, DoubleValue) {
  fp8_e5m3 v(4.0f);
  double out = static_cast<double>(v);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x88);
  EXPECT_EQ(out, 4.0);
}

TEST(FP8E5M3ConvertIntOp, CharValue) {
  fp8_e5m3 v(3.5f);
  char out = static_cast<char>(v);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x86);
  EXPECT_EQ(out, static_cast<char>(3));
}

TEST(FP8E5M3ConvertIntOp, SignedCharValue) {
  fp8_e5m3 v(6.5f);
  signed char out = static_cast<signed char>(v);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x8D);
  EXPECT_EQ(out, static_cast<signed char>(6));
}

TEST(FP8E5M3ConvertIntOp, ShortValue) {
  fp8_e5m3 v(7.5f);
  short out = static_cast<short>(v);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x8F);
  EXPECT_EQ(out, static_cast<short>(7));
}

TEST(FP8E5M3ConvertIntOp, IntValue) {
  fp8_e5m3 v(8.0f);
  int out = static_cast<int>(v);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x90);
  EXPECT_EQ(out, 8);
}

TEST(FP8E5M3ConvertIntOp, LongValue) {
  fp8_e5m3 v(9.0f);
  long out = static_cast<long>(v);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x91);
  EXPECT_EQ(out, 9L);
}

TEST(FP8E5M3ConvertIntOp, LongLongValue) {
  fp8_e5m3 v(10.0f);
  long long out = static_cast<long long>(v);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x92);
  EXPECT_EQ(out, 10LL);
}

TEST(FP8E5M3ConvertIntOp, UnsignedCharValue) {
  fp8_e5m3 v(11.0f);
  unsigned char out = static_cast<unsigned char>(v);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x93);
  EXPECT_EQ(out, static_cast<unsigned char>(11));
}

TEST(FP8E5M3ConvertIntOp, UnsignedShortValue) {
  fp8_e5m3 v(12.0f);
  unsigned short out = static_cast<unsigned short>(v);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x94);
  EXPECT_EQ(out, static_cast<unsigned short>(12));
}

TEST(FP8E5M3ConvertIntOp, UnsignedIntValue) {
  fp8_e5m3 v(13.0f);
  unsigned int out = static_cast<unsigned int>(v);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x95);
  EXPECT_EQ(out, 13u);
}

TEST(FP8E5M3ConvertIntOp, UnsignedLongValue) {
  fp8_e5m3 v(14.0f);
  unsigned long out = static_cast<unsigned long>(v);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x96);
  EXPECT_EQ(out, 14UL);
}

TEST(FP8E5M3ConvertIntOp, UnsignedLongLongValue) {
  fp8_e5m3 v(15.0f);
  unsigned long long out = static_cast<unsigned long long>(v);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x97);
  EXPECT_EQ(out, 15ULL);
}

TEST(FP8E5M3ConvertOp, BoolFalse) {
  fp8_e5m3 v(0.0f);
  bool out = static_cast<bool>(v);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x00);
  EXPECT_FALSE(out);
}

TEST(FP8E5M3ConvertOp, BoolTrue) {
  fp8_e5m3 v(0.25f);
  bool out = static_cast<bool>(v);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x68);
  EXPECT_TRUE(out);
}

TEST(FP8E5M3ConvertOp, MarrayHalf) {
  fp8_e5m3 v(1.625f);
  auto out = static_cast<sycl::marray<sycl::half, 1>>(v);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x7D);
  EXPECT_EQ(static_cast<float>(out[0]), 1.625f);
}

TEST(FP8E5M3ConvertOp, MarrayBfloat16) {
  fp8_e5m3 v(2.75f);
  auto out = static_cast<sycl::marray<bfloat16, 1>>(v);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x83);
  EXPECT_EQ(static_cast<float>(out[0]), 2.75f);
}

TEST(FP8E5M3ConvertOp, MarrayFloat) {
  fp8_e5m3 v(5.0f);
  auto out = static_cast<sycl::marray<float, 1>>(v);

  EXPECT_EQ(sizeof(v.vals), 1u);
  EXPECT_EQ(v.vals[0], 0x8A);
  EXPECT_EQ(out[0], 5.0f);
}
