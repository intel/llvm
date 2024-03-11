// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only %s
//
// This test is intended to check sycl::ext::oneapi::sub_group_mask interface.
// test for spec ver.2: sycl/test-e2e/SubGroupMask/sub_group_mask_ver2.cpp

#include <sycl/sycl.hpp>

#include <cstdint>
#include <type_traits>

int main() {
  using mask_type = sycl::ext::oneapi::sub_group_mask;
  using mask_reference_type = sycl::ext::oneapi::sub_group_mask &;

  auto mask = sycl::detail::Builder::createSubGroupMask<mask_type>(0, 32);
  const auto const_mask =
      sycl::detail::Builder::createSubGroupMask<mask_type>(0, 32);

  static_assert(
      std::is_same_v<decltype(mask[sycl::id(0)]), mask_type::reference>);
  static_assert(std::is_same_v<decltype(const_mask[sycl::id(0)]), bool>);

  static_assert(std::is_same_v<decltype(const_mask.test(sycl::id(0))), bool>);

  static_assert(std::is_same_v<decltype(const_mask.any()), bool>);
  static_assert(std::is_same_v<decltype(const_mask.all()), bool>);
  static_assert(std::is_same_v<decltype(const_mask.none()), bool>);

  static_assert(std::is_same_v<decltype(const_mask.count()), std::uint32_t>);
  static_assert(std::is_same_v<decltype(const_mask.size()), std::uint32_t>);

  static_assert(std::is_same_v<decltype(const_mask.find_low()), sycl::id<1>>);
  static_assert(std::is_same_v<decltype(const_mask.find_high()), sycl::id<1>>);

  int bits_i = 0;
  unsigned long long bits_ull = 0;
  sycl::marray<char, 3> bits_mc(0);
  sycl::marray<unsigned short, 12> bits_ms(0);

  mask.insert_bits(bits_i);
  mask.insert_bits(bits_ull, sycl::id(0));
  mask.insert_bits(bits_mc);
  mask.insert_bits(bits_ms, sycl::id(0));

  const_mask.extract_bits(bits_i);
  const_mask.extract_bits(bits_ull, sycl::id(0));
  const_mask.extract_bits(bits_mc);
  const_mask.extract_bits(bits_ms, sycl::id(0));

  mask.set();
  mask.set(sycl::id(0));
  mask.set(sycl::id(0), false);

  mask.reset();
  mask.reset(sycl::id(0));

  mask.reset_low();
  mask.reset_high();

  mask.flip();
  mask.flip(sycl::id(0));

  static_assert(std::is_same_v<decltype(const_mask == mask), bool>);
  static_assert(std::is_same_v<decltype(const_mask != mask), bool>);

  static_assert(
      std::is_same_v<decltype(mask &= const_mask), mask_reference_type>);
  static_assert(
      std::is_same_v<decltype(mask |= const_mask), mask_reference_type>);
  static_assert(
      std::is_same_v<decltype(mask ^= const_mask), mask_reference_type>);
  static_assert(std::is_same_v<decltype(mask <<= 3u), mask_reference_type>);
  static_assert(std::is_same_v<decltype(mask >>= 3u), mask_reference_type>);

  static_assert(std::is_same_v<decltype(~const_mask), mask_type>);
  static_assert(std::is_same_v<decltype(const_mask << 3u), mask_type>);
  static_assert(std::is_same_v<decltype(const_mask >> 3u), mask_type>);

  static_assert(std::is_same_v<decltype(const_mask & mask), mask_type>);
  static_assert(std::is_same_v<decltype(const_mask | mask), mask_type>);
  static_assert(std::is_same_v<decltype(const_mask ^ mask), mask_type>);

  return 0;
}
