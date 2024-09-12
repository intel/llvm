// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// RUN: %{build}  -fsycl-device-lib-jit-link -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: cuda || hip || cpu

#include "group_joint_KV_sort_p1p1_p1.hpp"
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
void test_work_group_KV_joint_sort(sycl::queue &q, KeyT keys[NUM],
                                   ValT vals[NUM], SortHelper gsh) {

  KeyT input_keys[NUM];
  ValT input_vals[NUM];
  memcpy((void *)input_keys, (void *)keys, NUM * sizeof(KeyT));
  memcpy((void *)input_vals, (void *)vals, NUM * sizeof(ValT));
  size_t scratch_size = NUM * (sizeof(KeyT) + sizeof(ValT)) +
                        std::max(alignof(KeyT), alignof(ValT));
  uint8_t *scratch_ptr =
      (uint8_t *)aligned_alloc_device(alignof(KeyT), scratch_size, q);
  const static size_t wg_size = WG_SZ;
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

  /*for (size_t idx = 0; idx < NUM; ++idx) {
    std::cout << "key: " << std::get<0>(sorted_vec[idx]) << " val: " <<
  std::get<1>(sorted_vec[idx]) << std::endl;
  }*/

  nd_range<1> num_items((range<1>(wg_size)), (range<1>(wg_size)));
  {
    buffer<KeyT, 1> ikeys_buf(input_keys, NUM);
    buffer<ValT, 1> ivals_buf(input_vals, NUM);
    q.submit([&](auto &h) {
       accessor ikeys_acc{ikeys_buf, h};
       accessor ivals_acc{ivals_buf, h};
       h.parallel_for(num_items, [=](nd_item<1> i) {
         gsh(ikeys_acc.template get_multi_ptr<access::decorated::no>().get(),
             ivals_acc.template get_multi_ptr<access::decorated::no>().get(),
             NUM, scratch_ptr);
       });
     }).wait();
  }

  /*for (size_t idx = 0; idx < NUM; ++idx) {
    std::cout << "key: " << (input_keys[idx]) << " val: " <<
  (input_vals[idx]) << std::endl;
  }*/

  sycl::free(scratch_ptr, q);
  bool fails = false;
  for (size_t idx = 0; idx < NUM; ++idx) {
    if ((input_keys[idx] != std::get<0>(sorted_vec[idx])) ||
        (input_vals[idx] != std::get<1>(sorted_vec[idx]))) {
      fails = true;
      std::cout << idx << std::endl;
      break;
    }
  }
  assert(!fails);
}

int main() {
  queue q;

  {
    constexpr static int NUM = 23;
    uint8_t ikeys[NUM] = {1,  11, 1,   9,  3,  100, 34, 8,  121, 77,  125, 23,
                          36, 2,  111, 91, 88, 2,   51, 91, 81,  122, 22};
    uint8_t ivals[NUM] = {99,  32,  1,   2,   67,  123, 253, 35,
                          111, 77,  165, 145, 254, 11,  161, 96,
                          165, 100, 64,  90,  255, 147, 135};
    int8_t ivals2[NUM] = {-1, 23, 0, 123, 99, 44, 8, 11, -67, -54, -113, 7,
                          1, 81, -81, 21, 25, -38, 66, 99, -121, 34, 45};

    auto work_group_sorter = [](uint8_t *keys, uint8_t *vals, uint32_t n,
                                uint8_t *scratch) {
#if defined(__SYCL_DEVICE_ONLY__) && (defined(__SPIR__) || defined(__SPIRV__))
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1u8_p1u8_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1u8_p1u8_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    auto work_group_sorter1 = [](uint8_t *keys, int8_t *vals, uint32_t n,
                                uint8_t *scratch) {
#if defined(__SYCL_DEVICE_ONLY__) && (defined(__SPIR__) || defined(__SPIRV__))
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1u8_p1i8_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1u8_p1i8_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    test_work_group_KV_joint_sort<uint8_t, uint8_t, 1, NUM,
                                  decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout << "KV joint sort p1u8_p1u8_u32_p1i8 <NUM = 23, WG = 1> pass."
              << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint8_t, 2, NUM,
                                  decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout << "KV joint sort p1u8_p1u8_u32_p1i8 <NUM = 23, WG = 2> pass."
              << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint8_t, 4, NUM,
                                  decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout << "KV joint sort p1u8_p1u8_u32_p1i8 <NUM = 23, WG = 4> pass."
              << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint8_t, 8, NUM,
                                  decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout << "KV joint sort p1u8_p1u8_u32_p1i8 <NUM = 23, WG = 8> pass."
              << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint8_t, 16, NUM,
                                  decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout << "KV joint sort p1u8_p1u8_u32_p1i8 <NUM = 23, WG = 16> pass."
              << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint8_t, 32, NUM,
                                  decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout << "KV joint sort p1u8_p1u8_u32_p1i8 <NUM = 23, WG = 32> pass."
              << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int8_t, 1, NUM,
                                  decltype(work_group_sorter1)>(
        q, ikeys, ivals2, work_group_sorter1);
    std::cout << "KV joint sort p1u8_p1i8_u32_p1i8 <NUM = 23, WG = 1> pass."
              << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int8_t, 2, NUM,
                                  decltype(work_group_sorter1)>(
        q, ikeys, ivals2, work_group_sorter1);
    std::cout << "KV joint sort p1u8_p1i8_u32_p1i8 <NUM = 23, WG = 2> pass."
              << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int8_t, 4, NUM,
                                  decltype(work_group_sorter1)>(
        q, ikeys, ivals2, work_group_sorter1);
    std::cout << "KV joint sort p1u8_p1i8_u32_p1i8 <NUM = 23, WG = 4> pass."
              << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int8_t, 8, NUM,
                                  decltype(work_group_sorter1)>(
        q, ikeys, ivals2, work_group_sorter1);
    std::cout << "KV joint sort p1u8_p1i8_u32_p1i8 <NUM = 23, WG = 8> pass."
              << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int8_t, 16, NUM,
                                  decltype(work_group_sorter1)>(
        q, ikeys, ivals2, work_group_sorter1);
    std::cout << "KV joint sort p1u8_p1i8_u32_p1i8 <NUM = 23, WG = 16> pass."
              << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int8_t, 32, NUM,
                                  decltype(work_group_sorter1)>(
        q, ikeys, ivals2, work_group_sorter1);
    std::cout << "KV joint sort p1u8_p1i8_u32_p1i8 <NUM = 23, WG = 32> pass."
              << std::endl;
  }

  {
    constexpr static int NUM = 21;
    uint32_t ikeys[NUM] = {1,  11, 1, 9,   3,  100, 34, 8,  121,   77, 125,
                           23, 36, 2, 111, 91, 88,  2,  51, 95431, 881};
    uint32_t ivals[NUM] = {99,  32,    1,   2,     67,   9123, 453,
                           435, 91111, 777, 165,   145,  2456, 88811,
                           761, 96,    765, 10000, 6364, 90,   525};
    auto work_group_sorter = [](uint32_t *keys, uint32_t *vals, uint32_t n,
                                uint8_t *scratch) {
#if defined(__SYCL_DEVICE_ONLY__) && (defined(__SPIR__) || defined(__SPIRV__))
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1u32_p1u32_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1u32_p1u32_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };
    test_work_group_KV_joint_sort<uint32_t, uint32_t, 16, NUM,
                                  decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout << "KV joint sort p1u32_p1u32_u32_p1i8 <NUM = 21, WG = 16> pass."
              << std::endl;
  }

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

#if defined(__SYCL_DEVICE_ONLY__) && (defined(__SPIR__) || defined(__SPIRV__))
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1u32_p1u32_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1u32_p1u32_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    test_work_group_KV_joint_sort<uint32_t, uint32_t, 2, NUM,
                                  decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout << "KV joint sort p1u32_p1u32_u32_p1i8 <NUM = 32, WG = 2> pass."
              << std::endl;

    test_work_group_KV_joint_sort<uint32_t, uint32_t, 4, NUM,
                                  decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout << "KV joint sort p1u32_p1u32_u32_p1i8 <NUM = 32, WG = 4> pass."
              << std::endl;

    test_work_group_KV_joint_sort<uint32_t, uint32_t, 8, NUM,
                                  decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout << "KV joint sort p1u32_p1u32_u32_p1i8 <NUM = 32, WG = 8> pass."
              << std::endl;

    test_work_group_KV_joint_sort<uint32_t, uint32_t, 16, NUM,
                                  decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout << "KV joint sort p1u32_p1u32_u32_p1i8 <NUM = 32, WG = 16> pass."
              << std::endl;

    test_work_group_KV_joint_sort<uint32_t, uint32_t, 8, NUM,
                                  decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout << "KV joint sort p1u32_p1u32_u32_p1i8 <NUM = 32, WG = 32> pass."
              << std::endl;
  }

  return 0;
}
