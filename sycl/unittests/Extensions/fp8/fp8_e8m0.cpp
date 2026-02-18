#include <gtest/gtest.h>
#include <sycl/ext/oneapi/experimental/float_8bit/types.hpp>

#include <cmath>
#include <cstdint>
#include <limits>

using namespace sycl::ext::oneapi::experimental;

TEST(FP8E8M0Test, VariadicConstructorFloat) {
	fp8_e8m0<4> a(1.0f, 2.0f, 1.1f, 0.0f);

	EXPECT_EQ(sizeof(a.vals), 4u);
	EXPECT_EQ(a.vals[0], 0x7F); // 1.0 -> exp=127
	EXPECT_EQ(a.vals[1], 0x80); // 2.0 -> exp=128
	EXPECT_EQ(a.vals[2], 0x80); // 1.1 -> upward to 2.0
	EXPECT_EQ(a.vals[3], 0x00); // 0.0 -> min normal
}

TEST(FP8E8M0Test, VariadicConstructorHalf) {
	fp8_e8m0<2> a(sycl::half(1.0f), sycl::half(3.0f));

	EXPECT_EQ(sizeof(a.vals), 2u);
	EXPECT_EQ(a.vals[0], 0x7F);
	EXPECT_EQ(a.vals[1], 0x81); // 3.0 -> upward to 4.0
}

TEST(FP8E8M0Test, VariadicConstructorBFloat16) {
	fp8_e8m0<2> a(sycl::ext::oneapi::bfloat16(1.0f),
							 sycl::ext::oneapi::bfloat16(2.0f));

	EXPECT_EQ(sizeof(a.vals), 2u);
	EXPECT_EQ(a.vals[0], 0x7F);
	EXPECT_EQ(a.vals[1], 0x80);
}

TEST(FP8E8M0Test, VariadicConstructorDouble) {
	fp8_e8m0<2> a(1.0, 3.0);

	EXPECT_EQ(sizeof(a.vals), 2u);
	EXPECT_EQ(a.vals[0], 0x7F);
	EXPECT_EQ(a.vals[1], 0x81);
}

TEST(FP8E8M0Test, VariadicConstructorBoundaryEncodings) {
	fp8_e8m0<3> a(std::ldexp(1.0f, 127), std::ldexp(1.0f, -127),
							 std::numeric_limits<float>::quiet_NaN());

	EXPECT_EQ(sizeof(a.vals), 3u);
	EXPECT_EQ(a.vals[0], 0xFE); // max normal
	EXPECT_EQ(a.vals[1], 0x00); // min normal
	EXPECT_EQ(a.vals[2], 0xFF); // NaN
}

TEST(FP8E8M0Test, CArrayConstructorFloatHostUpwardFinite) {
	const float in[5] = {1.0f, 1.1f, 3.0f, 0.0f, 1000.0f};
	fp8_e8m0<5> a(in, rounding::upward, saturation::finite);

	EXPECT_EQ(sizeof(a.vals), 5u);
	EXPECT_EQ(a.vals[0], 0x7F);
	EXPECT_EQ(a.vals[1], 0x80); // upward to 2.0
	EXPECT_EQ(a.vals[2], 0x81); // upward to 4.0
	EXPECT_EQ(a.vals[3], 0x00); // min normal
	EXPECT_EQ(a.vals[4], 0x89); // upward to 2^10 = 1024
}

TEST(FP8E8M0Test, CArrayConstructorHalfHostUpwardFinite) {
	const sycl::half in[4] = {sycl::half(1.0f), sycl::half(1.1f),
														sycl::half(3.0f), sycl::half(0.0f)};
	fp8_e8m0<4> a(in, rounding::upward, saturation::finite);

	EXPECT_EQ(sizeof(a.vals), 4u);
	EXPECT_EQ(a.vals[0], 0x7F);
	EXPECT_EQ(a.vals[1], 0x80);
	EXPECT_EQ(a.vals[2], 0x81);
	EXPECT_EQ(a.vals[3], 0x00);
}

TEST(FP8E8M0Test, CArrayConstructorBFloat16HostUpwardFinite) {
	const sycl::ext::oneapi::bfloat16 in[3] = {
			sycl::ext::oneapi::bfloat16(1.0f),
			sycl::ext::oneapi::bfloat16(2.0f),
			sycl::ext::oneapi::bfloat16(0.0f)};
	fp8_e8m0<3> a(in, rounding::upward, saturation::finite);

	EXPECT_EQ(sizeof(a.vals), 3u);
	EXPECT_EQ(a.vals[0], 0x7F);
	EXPECT_EQ(a.vals[1], 0x80);
	EXPECT_EQ(a.vals[2], 0x00);
}

TEST(FP8E8M0Test, CArrayConstructorDoubleDefaultUpwardFinite) {
	const double in[3] = {1.0, 3.0, 0.0};
	fp8_e8m0<3> a(in);

	EXPECT_EQ(sizeof(a.vals), 3u);
	EXPECT_EQ(a.vals[0], 0x7F);
	EXPECT_EQ(a.vals[1], 0x81);
	EXPECT_EQ(a.vals[2], 0x00);
}

TEST(FP8E8M0Test, MarrayConstructorAndOperatorsFloat) {
	sycl::marray<float, 4> in = {1.0f, 2.0f, 3.0f, 0.0f};
	fp8_e8m0<4> a(in, rounding::upward, saturation::finite);

	EXPECT_EQ(sizeof(a.vals), 4u);
	EXPECT_EQ(a.vals[0], 0x7F);
	EXPECT_EQ(a.vals[1], 0x80);
	EXPECT_EQ(a.vals[2], 0x81);
	EXPECT_EQ(a.vals[3], 0x00);

	sycl::marray<float, 4> out = static_cast<sycl::marray<float, 4>>(a);
	EXPECT_EQ(out[0], 1.0f);
	EXPECT_EQ(out[1], 2.0f);
	EXPECT_EQ(out[2], 4.0f);
	EXPECT_EQ(out[3], std::ldexp(1.0f, -127));
}

TEST(FP8E8M0Test, MarrayConstructorHalfBFloat16Double) {
	sycl::marray<sycl::half, 2> hvals = {sycl::half(1.0f), sycl::half(3.0f)};
	sycl::marray<sycl::ext::oneapi::bfloat16, 2> bvals = {
			sycl::ext::oneapi::bfloat16(1.0f),
			sycl::ext::oneapi::bfloat16(2.0f)};
	sycl::marray<double, 2> dvals = {1.0, 3.0};

	fp8_e8m0<2> ah(hvals, rounding::upward, saturation::finite);
	fp8_e8m0<2> ab(bvals, rounding::upward, saturation::finite);
	fp8_e8m0<2> ad(dvals);

	EXPECT_EQ(sizeof(ah.vals), 2u);
	EXPECT_EQ(sizeof(ab.vals), 2u);
	EXPECT_EQ(sizeof(ad.vals), 2u);

	EXPECT_EQ(ah.vals[0], 0x7F);
	EXPECT_EQ(ah.vals[1], 0x81);
	EXPECT_EQ(ab.vals[0], 0x7F);
	EXPECT_EQ(ab.vals[1], 0x80);
	EXPECT_EQ(ad.vals[0], 0x7F);
	EXPECT_EQ(ad.vals[1], 0x81);
}

TEST(FP8E8M0Test, IntegerConstructorsAllTypes) {
	fp8_e8m0<1> s(static_cast<short>(1));
	fp8_e8m0<1> i(static_cast<int>(2));
	fp8_e8m0<1> l(static_cast<long>(3));
	fp8_e8m0<1> ll(static_cast<long long>(4));
	fp8_e8m0<1> us(static_cast<unsigned short>(1));
	fp8_e8m0<1> ui(static_cast<unsigned int>(2));
	fp8_e8m0<1> ul(static_cast<unsigned long>(3));
	fp8_e8m0<1> ull(static_cast<unsigned long long>(4));

	EXPECT_EQ(sizeof(s.vals), 1u);
	EXPECT_EQ(sizeof(i.vals), 1u);
	EXPECT_EQ(sizeof(l.vals), 1u);
	EXPECT_EQ(sizeof(ll.vals), 1u);
	EXPECT_EQ(sizeof(us.vals), 1u);
	EXPECT_EQ(sizeof(ui.vals), 1u);
	EXPECT_EQ(sizeof(ul.vals), 1u);
	EXPECT_EQ(sizeof(ull.vals), 1u);

	EXPECT_EQ(s.vals[0], 0x7F);  // 1.0
	EXPECT_EQ(i.vals[0], 0x80);  // 2.0
	EXPECT_EQ(l.vals[0], 0x81);  // 3.0 -> upward to 4.0
	EXPECT_EQ(ll.vals[0], 0x81); // 4.0
	EXPECT_EQ(us.vals[0], 0x7F);
	EXPECT_EQ(ui.vals[0], 0x80);
	EXPECT_EQ(ul.vals[0], 0x81);
	EXPECT_EQ(ull.vals[0], 0x81);
}

TEST(FP8E8M0Test, AssignmentOperatorsAllTypes) {
	fp8_e8m0<1> a(1.0f);
	EXPECT_EQ(sizeof(a.vals), 1u);

	a = sycl::half(1.0f);
	EXPECT_EQ(a.vals[0], 0x7F);

	a = sycl::ext::oneapi::bfloat16(2.0f);
	EXPECT_EQ(a.vals[0], 0x80);

	a = 3.0f;
	EXPECT_EQ(a.vals[0], 0x81);

	a = 4.0;
	EXPECT_EQ(a.vals[0], 0x81);

	a = static_cast<short>(1);
	EXPECT_EQ(a.vals[0], 0x7F);

	a = static_cast<int>(2);
	EXPECT_EQ(a.vals[0], 0x80);

	a = static_cast<long>(3);
	EXPECT_EQ(a.vals[0], 0x81);

	a = static_cast<long long>(4);
	EXPECT_EQ(a.vals[0], 0x81);

	a = static_cast<unsigned short>(1);
	EXPECT_EQ(a.vals[0], 0x7F);

	a = static_cast<unsigned int>(2);
	EXPECT_EQ(a.vals[0], 0x80);

	a = static_cast<unsigned long>(3);
	EXPECT_EQ(a.vals[0], 0x81);

	a = static_cast<unsigned long long>(4);
	EXPECT_EQ(a.vals[0], 0x81);
}

TEST(FP8E8M0Test, FloatingPointConversionOperators) {
	fp8_e8m0<1> one(1.0f);
	fp8_e8m0<1> max(std::ldexp(1.0f, 127));
	fp8_e8m0<1> min(std::ldexp(1.0f, -127));

	EXPECT_EQ(sizeof(one.vals), 1u);
	EXPECT_EQ(one.vals[0], 0x7F);
	EXPECT_EQ(max.vals[0], 0xFE);
	EXPECT_EQ(min.vals[0], 0x00);

	float fo = static_cast<float>(one);
	double doo = static_cast<double>(one);
	sycl::half ho = static_cast<sycl::half>(one);
	sycl::ext::oneapi::bfloat16 bo = static_cast<sycl::ext::oneapi::bfloat16>(one);

	EXPECT_EQ(fo, 1.0f);
	EXPECT_EQ(doo, 1.0);
	EXPECT_EQ(static_cast<float>(ho), 1.0f);
	EXPECT_EQ(static_cast<float>(bo), 1.0f);

	sycl::half hmax = static_cast<sycl::half>(max);
	EXPECT_TRUE(std::isinf(static_cast<float>(hmax)));
	EXPECT_FALSE(std::signbit(static_cast<float>(hmax)));

	EXPECT_EQ(static_cast<float>(min), std::ldexp(1.0f, -127));
}

TEST(FP8E8M0Test, UnsignedConversionOperatorsTowardZero) {
	fp8_e8m0<1> a(3.0f); // upward to 4.0

	EXPECT_EQ(sizeof(a.vals), 1u);
	EXPECT_EQ(a.vals[0], 0x81);

	EXPECT_EQ(static_cast<unsigned char>(a), 4u);
	EXPECT_EQ(static_cast<unsigned short>(a), 4u);
	EXPECT_EQ(static_cast<unsigned int>(a), 4u);
	EXPECT_EQ(static_cast<unsigned long>(a), 4u);
	EXPECT_EQ(static_cast<unsigned long long>(a), 4u);
}

TEST(FP8E8M0Test, BoolOperatorAlwaysTrue) {
	fp8_e8m0<1> min(std::ldexp(1.0f, -127));
	fp8_e8m0<1> nanv(std::numeric_limits<float>::quiet_NaN());

	EXPECT_TRUE(static_cast<bool>(min));
	EXPECT_TRUE(static_cast<bool>(nanv));
}

TEST(FP8E8M0Test, MarrayConversionOperators) {
	fp8_e8m0<3> a(1.0f, 3.0f, std::ldexp(1.0f, 127));

	sycl::marray<sycl::half, 3> ho = static_cast<sycl::marray<sycl::half, 3>>(a);
	sycl::marray<sycl::ext::oneapi::bfloat16, 3> bo =
			static_cast<sycl::marray<sycl::ext::oneapi::bfloat16, 3>>(a);
	sycl::marray<float, 3> fo = static_cast<sycl::marray<float, 3>>(a);

	EXPECT_EQ(static_cast<float>(ho[0]), 1.0f);
	EXPECT_EQ(static_cast<float>(ho[1]), 4.0f);
	EXPECT_TRUE(std::isinf(static_cast<float>(ho[2])));

	EXPECT_EQ(static_cast<float>(bo[0]), 1.0f);
	EXPECT_EQ(static_cast<float>(bo[1]), 4.0f);
	EXPECT_EQ(static_cast<float>(bo[2]), std::ldexp(1.0f, 127));

	EXPECT_EQ(fo[0], 1.0f);
	EXPECT_EQ(fo[1], 4.0f);
	EXPECT_EQ(fo[2], std::ldexp(1.0f, 127));
}

