// Host and SPIR-V device.
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -I %sycl_include -Xclang -verify-ignore-unexpected=note,warning -Wno-sycl-strict -std=c++17
// expected-no-diagnostics
#include <sycl/sycl.hpp>

#include <cassert>

#if !defined(__SYCL_BITCAST_IS_CONSTEXPR)
#error "bfloat16 constexpr support requires __SYCL_BITCAST_IS_CONSTEXPR"
#endif

// sycl::half is only constexpr under these conditions (see half_type_impl.hpp).
#if defined(__SYCL_DEVICE_ONLY__) || __cpp_lib_bit_cast ||                     \
    (defined(__has_builtin) && __has_builtin(__builtin_bit_cast))
#define HALF_IS_CONSTEXPR 1
#else
#define HALF_IS_CONSTEXPR 0
#endif

using sycl::ext::oneapi::bfloat16;

// Constexpr tests for sycl::ext::oneapi::bfloat16. Operators that bfloat16.hpp
// defines with the same macro are only tested once.

// Operations without side effects work on constexpr locals, so they can be
// checked directly with static_assert.
void CheckNoSideEffects() {
  constexpr bfloat16 one{1.f};
  constexpr bfloat16 two{2.f};
  static_assert(static_cast<bool>(one),
                "bfloat16 to bool conversion must be constexpr");
  static_assert(static_cast<float>(one) == 1.f,
                "bfloat16 to float conversion must be constexpr");

  constexpr bfloat16 copy{one};
  static_assert(copy == 1.f, "bfloat16 copy constructor must be constexpr");

  static_assert(-one == -1.f, "bfloat16 unary minus must be constexpr");

  // Each binary operator has three overloads: bfloat16 with bfloat16, bfloat16
  // with T, and T with bfloat16.
  // + also covers -, * and /, which are defined by the same macro.
  static_assert(one + two == 3.f && one + 2 == 3.f && 2.0 + one == 3.f,
                "bfloat16 arithmetic operators must be constexpr");
  // == also covers !=, <, >, <= and >=, which are defined by the same macro.
  static_assert(one == one && !(one == two) && one == 1 && 1.0 == one,
                "bfloat16 comparison operators must be constexpr");

#if HALF_IS_CONSTEXPR
  constexpr sycl::half halfOne{1.f};
  constexpr bfloat16 fromHalf{halfOne};
  static_assert(fromHalf == 1.f,
                "bfloat16 constructor from sycl::half must be constexpr");
  static_assert(static_cast<sycl::half>(one) == halfOne,
                "bfloat16 to sycl::half conversion must be constexpr");
#endif
}

// Operations with side effects need mutable locals, which static_assert can't
// read inside the function. They are checked with assert instead: a failing
// assert can't be constant evaluated, so the static_assert below fails and
// the compiler points at the failing assert.
// All operations are applied in sequence to the same object.
constexpr bool CheckMutations() {
  bfloat16 bf16{1.f};
  const bfloat16 moved = std::move(bf16);
  assert(moved == 1.f && "bfloat16 move constructor must be constexpr");

  bf16 = 2.f;
  assert(bf16 == 2.f && "bfloat16 assignment from float must be constexpr");
  bf16 = moved;
  assert(bf16 == 1.f && "bfloat16 assignment from bfloat16 must be constexpr");

  // ++ also covers --, which is defined by the same macro.
  assert(++bf16 == 2.f && "bfloat16 pre-increment must be constexpr");
  const bfloat16 old = bf16++;
  assert(old == 2.f && bf16 == 3.f &&
         "bfloat16 post-increment must be constexpr");

  // += also covers -=, *= and /=, which are defined by the same macro.
  bf16 += moved;
  assert(bf16 == 4.f && "bfloat16 += must be constexpr");

#if HALF_IS_CONSTEXPR
  bf16 = sycl::half{2.f};
  assert(bf16 == 2.f &&
         "bfloat16 assignment from sycl::half must be constexpr");
#endif
  return true;
}

// Static assertion to trigger compile-time evaluation of CheckMutations()
// this assertion will never trigger, but if any of the asserts in
// CheckMutations() fail, this assertion will no longer have a constant value,
// causing a compile-time error.
static_assert(CheckMutations(),
              "bfloat16 mutating operations must be constexpr");

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
