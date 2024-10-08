// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/sub_group_mask.hpp>

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
#if SYCL_EXT_ONEAPI_SUB_GROUP_MASK >= 2
  using mask_type = sycl::ext::oneapi::sub_group_mask;

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
#else
  std::cout << "Test skipped due to unsupported extension." << std::endl;
#endif

  return 0;
}
