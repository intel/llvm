// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// RUN: %{build}  -fsycl-device-lib-jit-link -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: cuda || hip || cpu

#include "group_private_KV_sort_p1p1_p1.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <sycl.hpp>
#include <tuple>
#include <vector>

using namespace sycl;

template <typename KeyT, typename ValT, size_t WG_SZ, size_t NUM,
          typename SortHelper>
void test_work_group_KV_private_sort(sycl::queue &q, KeyT input_keys[NUM],
                                     ValT input_vals[NUM], SortHelper gsh) {
  static_assert((NUM % WG_SZ == 0),
                "Input number must be divisible by work group size!");
  size_t scratch_size = 2 * NUM * (sizeof(KeyT) + sizeof(ValT)) +
                        std::max(alignof(KeyT), alignof(ValT));
  uint8_t *scratch_ptr =
      (uint8_t *)aligned_alloc_device(alignof(KeyT), scratch_size, q);
  const static size_t wg_size = WG_SZ;
  constexpr size_t num_per_work_item = NUM / WG_SZ;
  KeyT output_keys[NUM];
  ValT output_vals[NUM];
  std::vector<std::tuple<KeyT, ValT>> sorted_vec;
  for (size_t idx = 0; idx < NUM; ++idx)
    sorted_vec.push_back(std::make_tuple(input_keys[idx], input_vals[idx]));
#ifdef DES
  auto kv_tuple_comp = [](const std::tuple<KeyT, ValT> &t1,
                          const std::tuple<KeyT, ValT> &t2) {
    return std::get<0>(t1) > std::get<0>(t2);
  };
#else
  auto kv_tuple_comp = [](const std::tuple<KeyT, ValT> &t1,
                          const std::tuple<KeyT, ValT> &t2) {
    return std::get<0>(t1) < std::get<0>(t2);
  };
#endif
  std::stable_sort(sorted_vec.begin(), sorted_vec.end(), kv_tuple_comp);

  nd_range<1> num_items((range<1>(wg_size)), (range<1>(wg_size)));
  {
    buffer<KeyT, 1> ikeys_buf(input_keys, NUM);
    buffer<ValT, 1> ivals_buf(input_vals, NUM);
    buffer<KeyT, 1> okeys_buf(output_keys, NUM);
    buffer<ValT, 1> ovals_buf(output_vals, NUM);
    q.submit([&](auto &h) {
       accessor ikeys_acc{ikeys_buf, h};
       accessor ivals_acc{ivals_buf, h};
       accessor okeys_acc{okeys_buf, h};
       accessor ovals_acc{ovals_buf, h};
       sycl::stream os(1024, 128, h);
       h.parallel_for(num_items, [=](nd_item<1> i) {
         KeyT pkeys[num_per_work_item];
         ValT pvals[num_per_work_item];
         // copy from global input to fix-size private array.
         for (size_t idx = 0; idx < num_per_work_item; ++idx) {
           pkeys[idx] =
               ikeys_acc[i.get_local_linear_id() * num_per_work_item + idx];
           pvals[idx] =
               ivals_acc[i.get_local_linear_id() * num_per_work_item + idx];
         }

         gsh(pkeys, pvals, num_per_work_item, scratch_ptr);

         for (size_t idx = 0; idx < num_per_work_item; ++idx) {
           okeys_acc[i.get_local_linear_id() * num_per_work_item + idx] =
               pkeys[idx];
           ovals_acc[i.get_local_linear_id() * num_per_work_item + idx] =
               pvals[idx];
         }
       });
     }).wait();
  }

  sycl::free(scratch_ptr, q);
  bool fails = false;
  for (size_t idx = 0; idx < NUM; ++idx) {
    if ((output_keys[idx] != std::get<0>(sorted_vec[idx])) ||
        (output_vals[idx] != std::get<1>(sorted_vec[idx]))) {
      std::cout << "idx: " << idx << std::endl;
      fails = true;
      break;
    }
  }
  assert(!fails);
}

int main() {
  queue q;

  {
    constexpr static int NUM = 32;
    uint32_t ikeys[NUM] = {1,   11,  1,   9,     3,    100,   34,   8,
                           121, 77,  125, 23,    36,   2,     111,  91,
                           88,  2,   51,  95431, 881,  99183, 31,   142,
                           416, 701, 699, 1024,  8912, 0,     7981, 17};
    uint32_t ivals[NUM] = {99,    32,    1,    2,   67,   9123,  453, 435,
                           91111, 777,   165,  145, 2456, 88811, 761, 96,
                           765,   10000, 6364, 90,  525,  882,   1,   2423,
                           9,     4324,  9123, 0,   1232, 777,   555, 314159};
    auto work_group_sorter = [](uint32_t *keys, uint32_t *vals, uint32_t n,
                                uint8_t *scratch) {
#ifdef DES
      __devicelib_default_work_group_private_sort_close_descending_p1u32_p1u32_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_ascending_p1u32_p1u32_u32_p1i8(
          keys, vals, n, scratch);
#endif
    };
    test_work_group_KV_private_sort<uint32_t, uint32_t, 8, NUM,
                                    decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout << "KV private sort p1u32_p1u32_u32_p1i8 pass." << std::endl;
  }

  {
    constexpr static int NUM = 35;
    uint8_t ikeys[NUM] = {1,   11,  1,   9,   3,  100, 34,  8,  121,
                          77,  125, 23,  36,  2,  111, 91,  88, 2,
                          51,  213, 181, 183, 31, 142, 216, 1,  199,
                          124, 12,  0,   181, 17, 15,  101, 44};
    uint32_t ivals[NUM] = {99,   32,    1,    2,      67,   9123, 453,
                           435,  91111, 777,  165,    145,  2456, 88811,
                           761,  96,    765,  10000,  6364, 90,   525,
                           882,  1,     2423, 9,      4324, 9123, 0,
                           1232, 777,   555,  314159, 905,  9831, 84341};
    auto work_group_sorter = [](uint8_t *keys, uint32_t *vals, uint32_t n,
                                uint8_t *scratch) {
#ifdef DES
      __devicelib_default_work_group_private_sort_close_descending_p1u8_p1u32_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_ascending_p1u8_p1u32_u32_p1i8(
          keys, vals, n, scratch);
#endif
    };
    test_work_group_KV_private_sort<uint8_t, uint32_t, 7, NUM,
                                    decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout << "KV private sort p1u8_p1u32_u32_p1i8 pass." << std::endl;
  }
}
