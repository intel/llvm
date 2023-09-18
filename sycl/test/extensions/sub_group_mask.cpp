// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out
//
// This test is intended to check sycl::ext::oneapi::sub_group_mask interface.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <bitset>
#include <climits>
#include <cstdint>
#include <type_traits>

class kernel;

#define TEST_ON_DEVICE(TEST_BODY)                                              \
  {                                                                            \
    sycl::queue queue;                                                         \
    sycl::buffer<bool, 1> buf(&res, 1);                                        \
    queue.submit([&](sycl::handler &h) {                                       \
      auto acc = buf.get_access<sycl::access_mode::write>(h);                  \
      h.parallel_for(sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),  \
                     [=](sycl::nd_item<1> item) { TEST_BODY });                \
    });                                                                        \
    queue.wait();                                                              \
  }                                                                            \
  assert(res);

int main() {
  using mask_type = sycl::ext::oneapi::sub_group_mask;
  using mask_reference_type = sycl::ext::oneapi::sub_group_mask &;

  {
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
    static_assert(
        std::is_same_v<decltype(const_mask.find_high()), sycl::id<1>>);

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
  }

// sycl_ext_oneapi_sub_group_mask rev.2
#if SYCL_EXT_ONEAPI_SUB_GROUP_MASK >= 2
  // sub_group_mask()
  {
    mask_type mask;
    assert(mask.none() && mask_type::max_bits == mask.size());

    bool res = false;
    // clang-format off
    TEST_ON_DEVICE(
      mask_type mask;
      auto sg = item.get_sub_group();
      acc[0] = mask.none() && (sg.get_max_local_range().size() == mask.size());
    )
    // clang-format on
  }
  // sub_group_mask(unsigned long long val)
  {
    unsigned long long val = 4815162342;
    mask_type mask(val);
    std::bitset<sizeof(val) * CHAR_BIT> bs(val);
    bool res = true;
    for (size_t i = 0;
         i < std::min(static_cast<size_t>(mask.size()), bs.size()); ++i)
      res &= mask[i] == bs[i];
    assert(res);

    // clang-format off
    TEST_ON_DEVICE(
      mask_type mask(val);
      auto sg = item.get_sub_group();
      acc[0] = sg.get_max_local_range().size() == mask.size();
      for (size_t i = 0;
            i < sycl::min(static_cast<size_t>(mask.size()), bs.size());
            ++i)
        acc[0] &= mask[i] == bs[i];
    )
    // clang-format on
  }
  // template <typename T, std::size_t K> sub_group_mask(const sycl::marray<T,
  // K>& &val)
  {
    sycl::marray<char, 4> marr{1, 2, 3, 4};
    mask_type mask(marr);
    std::bitset<CHAR_BIT> bs[4] = {1, 2, 3, 4};
    bool res = true;
    for (size_t i = 0; i < mask.size() && (i / CHAR_BIT) < 4; ++i)
      res &= mask[i] == bs[i / CHAR_BIT][i % CHAR_BIT];
    assert(res);

    // clang-format off
    TEST_ON_DEVICE(
      mask_type mask(marr);
      auto sg = item.get_sub_group();
      acc[0] = sg.get_max_local_range().size() == mask.size();
      for (size_t i = 0; i < mask.size() && (i / CHAR_BIT) < 4; ++i)
        acc[0] &= mask[i] == bs[i / CHAR_BIT][i % CHAR_BIT];
    )
    // clang-format on
  }
  {
    // sub_group_mask(const sub_group_mask &other)
    unsigned long long val = 4815162342;
    mask_type mask1(val);
    mask_type mask2(mask1);
    assert(mask1 == mask2);

    bool res = false;
    // clang-format off
    TEST_ON_DEVICE(
      mask_type mask1(val);
      mask_type mask2(mask1);
      acc[0] = mask1 == mask2;
    )
    // clang-format on
  }
  {
    // sub_group_mask& operator=(const sub_group_mask &other)
    unsigned long long val = 4815162342;
    mask_type mask1(val);
    mask_type mask2 = mask1;
    assert(mask1 == mask2);

    bool res = false;
    // clang-format off
    TEST_ON_DEVICE(
      mask_type mask1(val);
      mask_type mask2 = mask1;
      acc[0] = mask1 == mask2;
    )
    // clang-format on
  }
#endif

  return 0;
}
