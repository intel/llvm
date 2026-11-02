// Host and SPIR-V device.
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -I %sycl_include -Xclang -verify-ignore-unexpected=note,warning -Wno-sycl-strict -std=c++17
// expected-no-diagnostics
#include <sycl/sycl.hpp>

#if !defined(__SYCL_BITCAST_IS_CONSTEXPR)
#error "bfloat16 constexpr support requires __SYCL_BITCAST_IS_CONSTEXPR"
#endif

using sycl::ext::oneapi::bfloat16;

// Constexpr tests for sycl::ext::oneapi::bfloat16
constexpr bool CheckToBool() {
  const bfloat16 bf16{1.f};
  return static_cast<bool>(bf16);
}
static_assert(CheckToBool(), "bfloat16 to bool conversion must be constexpr");

constexpr float CheckToFloat() {
  const bfloat16 bf16{1.f};
  return bf16;
}
static_assert(CheckToFloat() == 1.f,
              "bfloat16 to float conversion must be constexpr");

constexpr float CheckAssignFloat() {
  bfloat16 bf16{1.f};
  bf16 = 2.f;
  return bf16;
}
static_assert(CheckAssignFloat() == 2.f,
              "bfloat16 assignment from float must be constexpr");

constexpr float CheckCtorBfloat16() {
  const bfloat16 a{1.f};
  const bfloat16 b{a};
  return b;
}
static_assert(CheckCtorBfloat16() == 1.f,
              "bfloat16 copy constructor must be constexpr");

constexpr float CheckMvCtorBfloat16() {
  bfloat16 a{1.f};
  const bfloat16 b = std::move(a);
  return b;
}
static_assert(CheckMvCtorBfloat16() == 1.f,
              "bfloat16 move constructor must be constexpr");

constexpr float CheckAssignBfloat16() {
  bfloat16 bf16{1.f};
  bf16 = bfloat16{2.f};
  return bf16;
}
static_assert(CheckAssignBfloat16() == 2.f,
              "bfloat16 assignment from bfloat16 must be constexpr");

constexpr float CheckNegate() {
  bfloat16 bf16{1.f};
  return -bf16;
}
static_assert(CheckNegate() == -1.f, "bfloat16 unary minus must be constexpr");

constexpr float CheckPreIncrement() {
  bfloat16 bf16{1.f};
  return ++bf16;
}
static_assert(CheckPreIncrement() == 2.f,
              "bfloat16 pre-increment must be constexpr");

constexpr float CheckPreDecrement() {
  bfloat16 bf16{2.f};
  return --bf16;
}
static_assert(CheckPreDecrement() == 1.f,
              "bfloat16 pre-decrement must be constexpr");

constexpr bool CheckPostIncrement() {
  bfloat16 bf16{1.f};
  const bfloat16 old = bf16++;
  return old == 1.f && bf16 == 2.f;
}
static_assert(CheckPostIncrement(),
              "bfloat16 post-increment must be constexpr");

constexpr bool CheckPostDecrement() {
  bfloat16 bf16{2.f};
  const bfloat16 old = bf16--;
  return old == 2.f && bf16 == 1.f;
}
static_assert(CheckPostDecrement(),
              "bfloat16 post-decrement must be constexpr");

constexpr float CheckAddAssign() {
  bfloat16 bf16{1.f};
  bf16 += bfloat16{2.f};
  return bf16;
}
static_assert(CheckAddAssign() == 3.f, "bfloat16 += must be constexpr");

constexpr float CheckSubAssign() {
  bfloat16 bf16{3.f};
  bf16 -= bfloat16{2.f};
  return bf16;
}
static_assert(CheckSubAssign() == 1.f, "bfloat16 -= must be constexpr");

constexpr float CheckMulAssign() {
  bfloat16 bf16{1.5f};
  bf16 *= bfloat16{2.f};
  return bf16;
}
static_assert(CheckMulAssign() == 3.f, "bfloat16 *= must be constexpr");

constexpr float CheckDivAssign() {
  bfloat16 bf16{3.f};
  bf16 /= bfloat16{2.f};
  return bf16;
}
static_assert(CheckDivAssign() == 1.5f, "bfloat16 /= must be constexpr");

// Each binary operator has three overloads: bfloat16 with bfloat16, bfloat16
// with T, and T with bfloat16.
constexpr bool CheckAdd() {
  const bfloat16 a{1.f};
  return a + bfloat16{2.f} == 3.f && a + 2 == 3.f && 2.0 + a == 3.f;
}
static_assert(CheckAdd(), "bfloat16 + must be constexpr");

constexpr bool CheckSub() {
  const bfloat16 a{3.f};
  return a - bfloat16{2.f} == 1.f && a - 2 == 1.f && 4.0 - a == 1.f;
}
static_assert(CheckSub(), "bfloat16 - must be constexpr");

constexpr bool CheckMul() {
  const bfloat16 a{1.5f};
  return a * bfloat16{2.f} == 3.f && a * 2 == 3.f && 2.0 * a == 3.f;
}
static_assert(CheckMul(), "bfloat16 * must be constexpr");

constexpr bool CheckDiv() {
  const bfloat16 a{3.f};
  return a / bfloat16{2.f} == 1.5f && a / 2 == 1.5f && 6.0 / a == 2.f;
}
static_assert(CheckDiv(), "bfloat16 / must be constexpr");

constexpr bool CheckEqual() {
  const bfloat16 a{1.f};
  return a == bfloat16{1.f} && !(a == bfloat16{2.f}) && a == 1 && 1.0 == a;
}
static_assert(CheckEqual(), "bfloat16 == must be constexpr");

constexpr bool CheckNotEqual() {
  const bfloat16 a{1.f};
  return a != bfloat16{2.f} && !(a != bfloat16{1.f}) && a != 2 && 2.0 != a;
}
static_assert(CheckNotEqual(), "bfloat16 != must be constexpr");

constexpr bool CheckLess() {
  const bfloat16 a{1.f};
  return a < bfloat16{2.f} && !(a < bfloat16{1.f}) && a < 2 && 0.0 < a;
}
static_assert(CheckLess(), "bfloat16 < must be constexpr");

constexpr bool CheckGreater() {
  const bfloat16 a{2.f};
  return a > bfloat16{1.f} && !(a > bfloat16{2.f}) && a > 1 && 3.0 > a;
}
static_assert(CheckGreater(), "bfloat16 > must be constexpr");

constexpr bool CheckLessEqual() {
  const bfloat16 a{1.f};
  return a <= bfloat16{1.f} && !(a <= bfloat16{0.f}) && a <= 2 && 1.0 <= a;
}
static_assert(CheckLessEqual(), "bfloat16 <= must be constexpr");

constexpr bool CheckGreaterEqual() {
  const bfloat16 a{1.f};
  return a >= bfloat16{1.f} && !(a >= bfloat16{2.f}) && a >= 0 && 1.0 >= a;
}
static_assert(CheckGreaterEqual(), "bfloat16 >= must be constexpr");

// Check the exact bits produced by the float to bfloat16 conversion, for
// values where comparing floats can't tell results apart.
constexpr uint16_t Bits(float f) {
  return sycl::bit_cast<uint16_t>(bfloat16{f});
}
static_assert(Bits(1.f) == 0x3f80, "1.0 must convert exactly");
static_assert(Bits(sycl::bit_cast<float>(0x3f808000u)) == 0x3f80,
              "a tie must round to even (down)");
static_assert(Bits(sycl::bit_cast<float>(0x3f818000u)) == 0x3f82,
              "a tie must round to even (up)");
static_assert(Bits(sycl::bit_cast<float>(0x3f808001u)) == 0x3f81,
              "above a tie must round up");
static_assert(Bits(0.1f) == 0x3dcd, "0.1 must round to nearest");
static_assert(Bits(__builtin_nanf("")) == 0xffc1, "NaN must convert to 0xffc1");
static_assert(Bits(__builtin_inff()) == 0x7f80, "+inf must convert exactly");
static_assert(Bits(-__builtin_inff()) == 0xff80, "-inf must convert exactly");
static_assert(Bits(sycl::bit_cast<float>(0x7f7fffffu)) == 0x7f80,
              "FLT_MAX must round to +inf");
static_assert(sycl::bit_cast<uint16_t>(-bfloat16{0.f}) == 0x8000,
              "negating 0 must give -0");

// sycl::half is only constexpr under these conditions (see half_type_impl.hpp).
#if defined(__SYCL_DEVICE_ONLY__) || __cpp_lib_bit_cast ||                     \
    (defined(__has_builtin) && __has_builtin(__builtin_bit_cast))
constexpr float CheckCtorHalf() {
  const sycl::half a{1.f};
  const bfloat16 b{a};
  return b;
}
static_assert(CheckCtorHalf() == 1.f,
              "bfloat16 constructor from sycl::half must be constexpr");

constexpr float CheckAssignHalf() {
  bfloat16 b{1.f};
  b = sycl::half{2.f};
  return b;
}
static_assert(CheckAssignHalf() == 2.f,
              "bfloat16 assignment from sycl::half must be constexpr");

constexpr sycl::half CheckToHalf() {
  const bfloat16 h{1.f};
  return static_cast<sycl::half>(h);
}
static_assert(CheckToHalf() == sycl::half{1.f},
              "bfloat16 to sycl::half conversion must be constexpr");
#endif
