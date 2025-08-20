/// Checks a numeric_limits specialization of bfloat16.

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/experimental/bfloat16_math.hpp>

namespace sycl_ext = sycl::ext::oneapi;

using Limit = std::numeric_limits<sycl_ext::bfloat16>;

// Result of std::log10(2).
constexpr float Log10_2 = 0.30103f;

// Helper constexpr ceil function.
constexpr int ceil(float Val) {
  return Val + (float(int(Val)) == Val ? 0.f : 1.f);
}

int Check(bool Condition, sycl_ext::bfloat16 Value, std::string CheckName) {
  if (!Condition)
    std::cout << "Failed " << CheckName << " for " << Value << std::endl;
  return !Condition;
}

int CheckBfloat16(uint16_t Sign, uint16_t Exponent, uint16_t Significand) {
  const auto Value = sycl::bit_cast<sycl_ext::bfloat16>(
      uint16_t((Sign << 15) | (Exponent << 7) | Significand));

  int Failed = 0;

  Failed += Check(Limit::lowest() <= Value, Value, "lowest()");
  Failed += Check(Limit::max() >= Value, Value, "max()");

  // min() is the lowest normal number, so if Value is negative, 0 or a
  // subnormal - the latter two being represented by a 0-exponent - min() must
  // be strictly greater.
  if (Sign || Exponent == 0x0)
    Failed += Check(Limit::min() > Value, Value, "min() (1)");
  else
    Failed += Check(Limit::min() <= Value, Value, "min() (2)");

  // denorm_min() is the lowest subnormal number, so if Value is negative or 0
  // denorm_min() must be strictly greater.
  if (Sign || (Exponent == 0x0 && Significand == 0x0))
    Failed += Check(Limit::denorm_min() > Value, Value, "denorm_min() (1)");
  else
    Failed += Check(Limit::denorm_min() <= Value, Value, "denorm_min() (2)");

  return Failed;
}

int main() {
  static_assert(Limit::is_specialized);
  static_assert(Limit::is_signed);
  static_assert(!Limit::is_integer);
  static_assert(!Limit::is_exact);
  static_assert(Limit::has_infinity);
  static_assert(Limit::has_quiet_NaN);
  static_assert(Limit::has_signaling_NaN);
  static_assert(Limit::has_denorm == std::float_denorm_style::denorm_present);
  static_assert(!Limit::has_denorm_loss);
  static_assert(!Limit::tinyness_before);
  static_assert(!Limit::traps);
  static_assert(Limit::max_exponent10 == 35);
  static_assert(Limit::max_exponent == 127);
  static_assert(Limit::min_exponent10 == -37);
  static_assert(Limit::min_exponent == -126);
  static_assert(Limit::radix == 2);
  static_assert(Limit::digits == 8);
  static_assert(Limit::max_digits10 ==
                ceil(float(Limit::digits) * Log10_2 + 1.0f));
  static_assert(Limit::is_bounded);
  static_assert(Limit::digits10 == int(Limit::digits * Log10_2));
  static_assert(!Limit::is_modulo);
  static_assert(Limit::is_iec559);
  static_assert(Limit::round_style == std::float_round_style::round_to_nearest);

  int Failed = 0;

  Failed += Check(sycl_ext::experimental::isnan(Limit::quiet_NaN()),
                  Limit::quiet_NaN(), "quiet_NaN()");
  Failed += Check(sycl_ext::experimental::isnan(Limit::signaling_NaN()),
                  Limit::signaling_NaN(), "signaling_NaN()");
  // isinf does not exist for bfloat16 currently.
  Failed += Check(Limit::infinity() ==
                      sycl::bit_cast<sycl_ext::bfloat16>(uint16_t(0xff << 7)),
                  Limit::infinity(), "infinity()");
  Failed += Check(Limit::round_error() == sycl_ext::bfloat16(0.5f),
                  Limit::round_error(), "round_error()");
  Failed += Check(sycl_ext::bfloat16{1.0f} + Limit::epsilon() >
                      sycl_ext::bfloat16{1.0f},
                  Limit::epsilon(), "epsilon()");

  for (uint16_t Sign : {0, 1})
    for (uint16_t Exponent = 0; Exponent < 0xff; ++Exponent)
      for (uint16_t Significand = 0; Significand < 0x7f; ++Significand)
        Failed += CheckBfloat16(Sign, Exponent, Significand);

  return Failed;
}
