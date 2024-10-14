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
#include <sycl/sycl.hpp>
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

  /* for (size_t idx = 0; idx < NUM; ++idx) {
    std::cout << "key: " << (int)std::get<0>(sorted_vec[idx]) << " val: " <<
  (int)std::get<1>(sorted_vec[idx]) << std::endl;
  } */

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

  /* for (size_t idx = 0; idx < NUM; ++idx) {
    std::cout << "key: " << (int)(input_keys[idx]) << " val: " <<
  (int)(input_vals[idx]) << std::endl;
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
    int8_t ikeys1[NUM] = {0,  1,  -2, 1, 122, -123, 99,   -91, 9,   12, 13, 46,
                          13, 13, 9,  5, 77,  81,   -100, 35,  -64, 22, 23};
    uint8_t ivals[NUM] = {99,  32,  1,   2,   67,  123, 253, 35,
                          111, 77,  165, 145, 254, 11,  161, 96,
                          165, 100, 64,  90,  255, 147, 135};
    int8_t ivals2[NUM] = {-1, 23, 0,   123, 99, 44,  8,  11, -67,  -54, -113, 7,
                          1,  81, -81, 21,  25, -38, 66, 99, -121, 34,  45};

    uint16_t ivals3[NUM] = {36882, 47565, 20664, 59517, 55773, 5858,
                            30720, 64786, 42129, 13618, 62202, 16225,
                            54751, 38268, 25563, 44332, 45475, 12550,
                            5478,  3301,  3779,  25518, 6659};

    int16_t ivals4[NUM] = {3882,   7565, -20664, 9517,  -5773, 5858,
                           -30720, 86,   429,    13618, 2202,  -16225,
                           751,    -368, 25563,  -4332, -5475, 12550,
                           5478,   3301, 3779,   25518, 6659};

    uint32_t ivals5[NUM] = {
        2,          771,       76,        450,        76421894, 273377,
        85040,      831870667, 402825730, 2774821,    10786,    47164,
        1951118976, 75033606,  35755,     93312,      21,       3257266819,
        1065029990, 139884,    11355,     1464548796, 403290};

    int32_t ivals6[NUM] = {
        2,           771,       -76,        450,        76421894, 273377,
        85040,       831870667, -402825730, -2774821,   10786,    47164,
        1951118976,  -75033606, 35755,      93312,      21,       57266819,
        -1065029990, 139884,    -11355,     1464548796, -403290};

    uint64_t ivals7[NUM] = {
        2,          771,        76,          1112450,        76421894,
        898273377,  66585040,   11831870667, 402825730,      2774821,
        10786,      47164,      1951118976,  75033606,       99935755,
        9331211,    21,         3257266819,  10650299901112, 1224139884,
        9837411355, 1464548796, 403290};

    int64_t ivals8[NUM] = {
        2,           771,        76,           1112450,         76421894,
        898273377,   66585040,   -11831870667, -402825730,      2774821,
        10786,       47164,      1951118976,   75033606,        10,
        0,           21,         -3257266819,  -10650299901112, 76,
        -9837411355, 1464548796, 403290};

    float ivals9[NUM] = {
        1.628561f,  2.998057f,  0.082604f,    0.0f,        12.330798f,
        -1.350443f, 0.437885f,  0.017387f,    0.474454f,   -0.718838f,
        98.150388f, 0.732236f,  0.519963f,    -0.332644f,  0.648420f,
        0.578913f,  -0.853190f, -910.141650f, 110.037210f, 0.434222f,
        -0.343777f, 0.346011f,  0.767590f,
    };

    auto work_group_sorter = [](uint8_t *keys, uint8_t *vals, uint32_t n,
                                uint8_t *scratch) {
#if __DEVICE_CODE
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
#if __DEVICE_CODE
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1u8_p1i8_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1u8_p1i8_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    auto work_group_sorter2 = [](uint8_t *keys, uint16_t *vals, uint32_t n,
                                 uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1u8_p1u16_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1u8_p1u16_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    auto work_group_sorter3 = [](uint8_t *keys, int16_t *vals, uint32_t n,
                                 uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1u8_p1i16_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1u8_p1i16_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    auto work_group_sorter4 = [](uint8_t *keys, uint32_t *vals, uint32_t n,
                                 uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1u8_p1u32_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1u8_p1u32_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    auto work_group_sorter5 = [](uint8_t *keys, int32_t *vals, uint32_t n,
                                 uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1u8_p1i32_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1u8_p1i32_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    auto work_group_sorter6 = [](uint8_t *keys, uint64_t *vals, uint32_t n,
                                 uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1u8_p1u64_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1u8_p1u64_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    auto work_group_sorter7 = [](uint8_t *keys, int64_t *vals, uint32_t n,
                                 uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1u8_p1i64_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1u8_p1i64_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    auto work_group_sorter8 = [](uint8_t *keys, float *vals, uint32_t n,
                                 uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1u8_p1f32_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1u8_p1f32_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    auto work_group_sorter9 = [](int8_t *keys, uint8_t *vals, uint32_t n,
                                 uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1i8_p1u8_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1i8_p1u8_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    auto work_group_sorter10 = [](int8_t *keys, uint16_t *vals, uint32_t n,
                                  uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1i8_p1u16_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1i8_p1u16_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    auto work_group_sorter11 = [](int8_t *keys, uint32_t *vals, uint32_t n,
                                  uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1i8_p1u32_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1i8_p1u32_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    auto work_group_sorter12 = [](int8_t *keys, uint64_t *vals, uint32_t n,
                                  uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1i8_p1u64_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1i8_p1u64_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    auto work_group_sorter13 = [](int8_t *keys, int8_t *vals, uint32_t n,
                                  uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1i8_p1i8_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1i8_p1i8_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    auto work_group_sorter14 = [](int8_t *keys, int16_t *vals, uint32_t n,
                                  uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1i8_p1i16_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1i8_p1i16_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    test_work_group_KV_joint_sort<int8_t, uint8_t, 1, NUM,
                                  decltype(work_group_sorter9)>(
        q, ikeys1, ivals, work_group_sorter9);
    std::cout
        << "KV joint sort (Key: int8_t, Val: uint8_t) <NUM = 23, WG = 1> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, uint8_t, 2, NUM,
                                  decltype(work_group_sorter9)>(
        q, ikeys1, ivals, work_group_sorter9);
    std::cout
        << "KV joint sort (Key: int8_t, Val: uint8_t) <NUM = 23, WG = 2> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, uint8_t, 4, NUM,
                                  decltype(work_group_sorter9)>(
        q, ikeys1, ivals, work_group_sorter9);
    std::cout
        << "KV joint sort (Key: int8_t, Val: uint8_t) <NUM = 23, WG = 4> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, uint8_t, 8, NUM,
                                  decltype(work_group_sorter9)>(
        q, ikeys1, ivals, work_group_sorter9);
    std::cout
        << "KV joint sort (Key: int8_t, Val: uint8_t) <NUM = 23, WG = 8> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, uint8_t, 16, NUM,
                                  decltype(work_group_sorter9)>(
        q, ikeys1, ivals, work_group_sorter9);
    std::cout
        << "KV joint sort (Key: int8_t, Val: uint8_t) <NUM = 23, WG = 16> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, uint8_t, 32, NUM,
                                  decltype(work_group_sorter9)>(
        q, ikeys1, ivals, work_group_sorter9);
    std::cout
        << "KV joint sort (Key: int8_t, Val: uint8_t) <NUM = 23, WG = 32> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, int8_t, 1, NUM,
                                  decltype(work_group_sorter13)>(
        q, ikeys1, ivals2, work_group_sorter13);
    std::cout
        << "KV joint sort (Key: int8_t, Val: int8_t) <NUM = 23, WG = 1> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, int8_t, 2, NUM,
                                  decltype(work_group_sorter13)>(
        q, ikeys1, ivals2, work_group_sorter13);
    std::cout
        << "KV joint sort (Key: int8_t, Val: int8_t) <NUM = 23, WG = 2> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, int8_t, 4, NUM,
                                  decltype(work_group_sorter13)>(
        q, ikeys1, ivals2, work_group_sorter13);
    std::cout
        << "KV joint sort (Key: int8_t, Val: int8_t) <NUM = 23, WG = 4> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, int8_t, 8, NUM,
                                  decltype(work_group_sorter13)>(
        q, ikeys1, ivals2, work_group_sorter13);
    std::cout
        << "KV joint sort (Key: int8_t, Val: int8_t) <NUM = 23, WG = 8> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, int8_t, 16, NUM,
                                  decltype(work_group_sorter13)>(
        q, ikeys1, ivals2, work_group_sorter13);
    std::cout
        << "KV joint sort (Key: int8_t, Val: int8_t) <NUM = 23, WG = 16> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, int8_t, 32, NUM,
                                  decltype(work_group_sorter13)>(
        q, ikeys1, ivals2, work_group_sorter13);
    std::cout
        << "KV joint sort (Key: int8_t, Val: int8_t) <NUM = 23, WG = 32> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, uint16_t, 1, NUM,
                                  decltype(work_group_sorter10)>(
        q, ikeys1, ivals3, work_group_sorter10);
    std::cout
        << "KV joint sort (Key: int8_t, Val: uint16_t) <NUM = 23, WG = 1> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, uint16_t, 2, NUM,
                                  decltype(work_group_sorter10)>(
        q, ikeys1, ivals3, work_group_sorter10);
    std::cout
        << "KV joint sort (Key: int8_t, Val: uint16_t) <NUM = 23, WG = 2> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, uint16_t, 4, NUM,
                                  decltype(work_group_sorter10)>(
        q, ikeys1, ivals3, work_group_sorter10);
    std::cout
        << "KV joint sort (Key: int8_t, Val: uint16_t) <NUM = 23, WG = 4> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, uint16_t, 8, NUM,
                                  decltype(work_group_sorter10)>(
        q, ikeys1, ivals3, work_group_sorter10);
    std::cout
        << "KV joint sort (Key: int8_t, Val: uint16_t) <NUM = 23, WG = 8> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, uint16_t, 16, NUM,
                                  decltype(work_group_sorter10)>(
        q, ikeys1, ivals3, work_group_sorter10);
    std::cout << "KV joint sort (Key: int8_t, Val: uint16_t) <NUM = 23, WG = "
                 "16> pass."
              << std::endl;

    test_work_group_KV_joint_sort<int8_t, uint16_t, 32, NUM,
                                  decltype(work_group_sorter10)>(
        q, ikeys1, ivals3, work_group_sorter10);
    std::cout << "KV joint sort (Key: int8_t, Val: uint16_t) <NUM = 23, WG = "
                 "32> pass."
              << std::endl;

    test_work_group_KV_joint_sort<int8_t, int16_t, 1, NUM,
                                  decltype(work_group_sorter14)>(
        q, ikeys1, ivals4, work_group_sorter14);
    std::cout
        << "KV joint sort (Key: int8_t, Val: int16_t) <NUM = 23, WG = 1> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, int16_t, 2, NUM,
                                  decltype(work_group_sorter14)>(
        q, ikeys1, ivals4, work_group_sorter14);
    std::cout
        << "KV joint sort (Key: int8_t, Val: int16_t) <NUM = 23, WG = 2> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, int16_t, 4, NUM,
                                  decltype(work_group_sorter14)>(
        q, ikeys1, ivals4, work_group_sorter14);
    std::cout
        << "KV joint sort (Key: int8_t, Val: int16_t) <NUM = 23, WG = 4> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, int16_t, 8, NUM,
                                  decltype(work_group_sorter14)>(
        q, ikeys1, ivals4, work_group_sorter14);
    std::cout
        << "KV joint sort (Key: int8_t, Val: int16_t) <NUM = 23, WG = 8> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, int16_t, 16, NUM,
                                  decltype(work_group_sorter14)>(
        q, ikeys1, ivals4, work_group_sorter14);
    std::cout
        << "KV joint sort (Key: int8_t, Val: int16_t) <NUM = 23, WG = 16> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, int16_t, 32, NUM,
                                  decltype(work_group_sorter14)>(
        q, ikeys1, ivals4, work_group_sorter14);
    std::cout
        << "KV joint sort (Key: int8_t, Val: int16_t) <NUM = 23, WG = 32> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, uint32_t, 1, NUM,
                                  decltype(work_group_sorter11)>(
        q, ikeys1, ivals5, work_group_sorter11);
    std::cout
        << "KV joint sort (Key: int8_t, Val: uint32_t) <NUM = 23, WG = 1> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, uint32_t, 2, NUM,
                                  decltype(work_group_sorter11)>(
        q, ikeys1, ivals5, work_group_sorter11);
    std::cout
        << "KV joint sort (Key: int8_t, Val: uint32_t) <NUM = 23, WG = 2> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, uint32_t, 4, NUM,
                                  decltype(work_group_sorter11)>(
        q, ikeys1, ivals5, work_group_sorter11);
    std::cout
        << "KV joint sort (Key: int8_t, Val: uint32_t) <NUM = 23, WG = 4> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, uint32_t, 8, NUM,
                                  decltype(work_group_sorter11)>(
        q, ikeys1, ivals5, work_group_sorter11);
    std::cout
        << "KV joint sort (Key: int8_t, Val: uint32_t) <NUM = 23, WG = 8> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, uint32_t, 16, NUM,
                                  decltype(work_group_sorter11)>(
        q, ikeys1, ivals5, work_group_sorter11);
    std::cout << "KV joint sort (Key: int8_t, Val: uint32_t) <NUM = 23, WG = "
                 "16> pass."
              << std::endl;

    test_work_group_KV_joint_sort<int8_t, uint32_t, 32, NUM,
                                  decltype(work_group_sorter11)>(
        q, ikeys1, ivals5, work_group_sorter11);
    std::cout << "KV joint sort (Key: int8_t, Val: uint32_t) <NUM = 23, WG = "
                 "32> pass."
              << std::endl;

    test_work_group_KV_joint_sort<int8_t, uint64_t, 1, NUM,
                                  decltype(work_group_sorter12)>(
        q, ikeys1, ivals7, work_group_sorter12);
    std::cout
        << "KV joint sort (Key: int8_t, Val: uint64_t) <NUM = 23, WG = 1> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, uint64_t, 2, NUM,
                                  decltype(work_group_sorter12)>(
        q, ikeys1, ivals7, work_group_sorter12);
    std::cout
        << "KV joint sort (Key: int8_t, Val: uint64_t) <NUM = 23, WG = 2> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, uint64_t, 4, NUM,
                                  decltype(work_group_sorter12)>(
        q, ikeys1, ivals7, work_group_sorter12);
    std::cout
        << "KV joint sort (Key: int8_t, Val: uint64_t) <NUM = 23, WG = 4> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, uint64_t, 8, NUM,
                                  decltype(work_group_sorter12)>(
        q, ikeys1, ivals7, work_group_sorter12);
    std::cout
        << "KV joint sort (Key: int8_t, Val: uint64_t) <NUM = 23, WG = 8> pass."
        << std::endl;

    test_work_group_KV_joint_sort<int8_t, uint64_t, 16, NUM,
                                  decltype(work_group_sorter12)>(
        q, ikeys1, ivals7, work_group_sorter12);
    std::cout << "KV joint sort (Key: int8_t, Val: uint64_t) <NUM = 23, WG = "
                 "16> pass."
              << std::endl;

    test_work_group_KV_joint_sort<int8_t, uint64_t, 32, NUM,
                                  decltype(work_group_sorter12)>(
        q, ikeys1, ivals7, work_group_sorter12);
    std::cout << "KV joint sort (Key: int8_t, Val: uint64_t) <NUM = 23, WG = "
                 "32> pass."
              << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint8_t, 1, NUM,
                                  decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout
        << "KV joint sort (Key: uint8_t, Val: uint8_t) <NUM = 23, WG = 1> pass."
        << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint8_t, 2, NUM,
                                  decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout
        << "KV joint sort (Key: uint8_t, Val: uint8_t) <NUM = 23, WG = 2> pass."
        << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint8_t, 4, NUM,
                                  decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout
        << "KV joint sort (Key: uint8_t, Val: uint8_t) <NUM = 23, WG = 4> pass."
        << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint8_t, 8, NUM,
                                  decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout
        << "KV joint sort (Key: uint8_t, Val: uint8_t) <NUM = 23, WG = 8> pass."
        << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint8_t, 16, NUM,
                                  decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout << "KV joint sort (Key: uint8_t, Val: uint8_t) <NUM = 23, WG = "
                 "16> pass."
              << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint8_t, 32, NUM,
                                  decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout << "KV joint sort (Key: uint8_t, Val: uint8_t) <NUM = 23, WG = "
                 "32> pass."
              << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int8_t, 1, NUM,
                                  decltype(work_group_sorter1)>(
        q, ikeys, ivals2, work_group_sorter1);
    std::cout
        << "KV joint sort (Key: uint8_t, Val: int8_t) <NUM = 23, WG = 1> pass."
        << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int8_t, 2, NUM,
                                  decltype(work_group_sorter1)>(
        q, ikeys, ivals2, work_group_sorter1);
    std::cout
        << "KV joint sort (Key: uint8_t, Val: int8_t) <NUM = 23, WG = 2> pass."
        << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int8_t, 4, NUM,
                                  decltype(work_group_sorter1)>(
        q, ikeys, ivals2, work_group_sorter1);
    std::cout
        << "KV joint sort (Key: uint8_t, Val: int8_t) <NUM = 23, WG = 4> pass."
        << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int8_t, 8, NUM,
                                  decltype(work_group_sorter1)>(
        q, ikeys, ivals2, work_group_sorter1);
    std::cout
        << "KV joint sort (Key: uint8_t, Val: int8_t) <NUM = 23, WG = 8> pass."
        << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int8_t, 16, NUM,
                                  decltype(work_group_sorter1)>(
        q, ikeys, ivals2, work_group_sorter1);
    std::cout
        << "KV joint sort (Key: uint8_t, Val: int8_t) <NUM = 23, WG = 16> pass."
        << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int8_t, 32, NUM,
                                  decltype(work_group_sorter1)>(
        q, ikeys, ivals2, work_group_sorter1);
    std::cout
        << "KV joint sort (Key: uint8_t, Val: int8_t) <NUM = 23, WG = 32> pass."
        << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint16_t, 1, NUM,
                                  decltype(work_group_sorter2)>(
        q, ikeys, ivals3, work_group_sorter2);
    std::cout << "KV joint sort (Key: uint8_t, Val: uint16_t) <NUM =" << NUM
              << ", WG = 1> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint16_t, 2, NUM,
                                  decltype(work_group_sorter2)>(
        q, ikeys, ivals3, work_group_sorter2);
    std::cout << "KV joint sort (Key: uint8_t, Val: uint16_t) <NUM =" << NUM
              << ", WG = 2> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint16_t, 4, NUM,
                                  decltype(work_group_sorter2)>(
        q, ikeys, ivals3, work_group_sorter2);
    std::cout << "KV joint sort (Key: uint8_t, Val: uint16_t) <NUM =" << NUM
              << ", WG = 4> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint16_t, 8, NUM,
                                  decltype(work_group_sorter2)>(
        q, ikeys, ivals3, work_group_sorter2);
    std::cout << "KV joint sort (Key: uint8_t, Val: uint16_t) <NUM =" << NUM
              << ", WG = 8> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint16_t, 16, NUM,
                                  decltype(work_group_sorter2)>(
        q, ikeys, ivals3, work_group_sorter2);
    std::cout << "KV joint sort (Key: uint8_t, Val: uint16_t) <NUM =" << NUM
              << ", WG = 16> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint16_t, 32, NUM,
                                  decltype(work_group_sorter2)>(
        q, ikeys, ivals3, work_group_sorter2);
    std::cout << "KV joint sort (Key: uint8_t, Val: uint16_t) <NUM =" << NUM
              << ", WG = 32> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int16_t, 1, NUM,
                                  decltype(work_group_sorter3)>(
        q, ikeys, ivals4, work_group_sorter3);
    std::cout << "KV joint sort (Key: uint8_t, Val: int16_t) <NUM =" << NUM
              << ", WG = 1> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int16_t, 2, NUM,
                                  decltype(work_group_sorter3)>(
        q, ikeys, ivals4, work_group_sorter3);
    std::cout << "KV joint sort (Key: uint8_t, Val: int16_t) <NUM =" << NUM
              << ", WG = 2> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int16_t, 4, NUM,
                                  decltype(work_group_sorter3)>(
        q, ikeys, ivals4, work_group_sorter3);
    std::cout << "KV joint sort (Key: uint8_t, Val: int16_t) <NUM =" << NUM
              << ", WG = 4> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int16_t, 8, NUM,
                                  decltype(work_group_sorter3)>(
        q, ikeys, ivals4, work_group_sorter3);
    std::cout << "KV joint sort (Key: uint8_t, Val: int16_t) <NUM =" << NUM
              << ", WG = 8> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int16_t, 16, NUM,
                                  decltype(work_group_sorter3)>(
        q, ikeys, ivals4, work_group_sorter3);
    std::cout << "KV joint sort (Key: uint8_t, Val: int16_t) <NUM =" << NUM
              << ", WG = 16> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int16_t, 32, NUM,
                                  decltype(work_group_sorter3)>(
        q, ikeys, ivals4, work_group_sorter3);
    std::cout << "KV joint sort (Key: uint8_t, Val: int16_t) <NUM =" << NUM
              << ", WG = 32> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint32_t, 1, NUM,
                                  decltype(work_group_sorter4)>(
        q, ikeys, ivals5, work_group_sorter4);
    std::cout << "KV joint sort (Key: uint8_t, Val: uint32_t) <NUM =" << NUM
              << ", WG = 1> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint32_t, 2, NUM,
                                  decltype(work_group_sorter4)>(
        q, ikeys, ivals5, work_group_sorter4);
    std::cout << "KV joint sort (Key: uint8_t, Val: uint32_t) <NUM =" << NUM
              << ", WG = 2> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint32_t, 4, NUM,
                                  decltype(work_group_sorter4)>(
        q, ikeys, ivals5, work_group_sorter4);
    std::cout << "KV joint sort (Key: uint8_t, Val: uint32_t) <NUM =" << NUM
              << ", WG = 4> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint32_t, 8, NUM,
                                  decltype(work_group_sorter4)>(
        q, ikeys, ivals5, work_group_sorter4);
    std::cout << "KV joint sort (Key: uint8_t, Val: uint32_t) <NUM =" << NUM
              << ", WG = 8> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint32_t, 16, NUM,
                                  decltype(work_group_sorter4)>(
        q, ikeys, ivals5, work_group_sorter4);
    std::cout << "KV joint sort (Key: uint8_t, Val: uint32_t) <NUM =" << NUM
              << ", WG = 16> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint32_t, 32, NUM,
                                  decltype(work_group_sorter4)>(
        q, ikeys, ivals5, work_group_sorter4);
    std::cout << "KV joint sort (Key: uint8_t, Val: uint32_t) <NUM =" << NUM
              << ", WG = 32> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int32_t, 1, NUM,
                                  decltype(work_group_sorter5)>(
        q, ikeys, ivals6, work_group_sorter5);
    std::cout << "KV joint sort (Key: uint8_t, Val: int32_t) <NUM =" << NUM
              << ", WG = 1> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int32_t, 2, NUM,
                                  decltype(work_group_sorter5)>(
        q, ikeys, ivals6, work_group_sorter5);
    std::cout << "KV joint sort (Key: uint8_t, Val: int32_t) <NUM =" << NUM
              << ", WG = 2> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int32_t, 4, NUM,
                                  decltype(work_group_sorter5)>(
        q, ikeys, ivals6, work_group_sorter5);
    std::cout << "KV joint sort (Key: uint8_t, Val: int32_t) <NUM =" << NUM
              << ", WG = 4> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int32_t, 8, NUM,
                                  decltype(work_group_sorter5)>(
        q, ikeys, ivals6, work_group_sorter5);
    std::cout << "KV joint sort (Key: uint8_t, Val: int32_t) <NUM =" << NUM
              << ", WG = 8> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int32_t, 16, NUM,
                                  decltype(work_group_sorter5)>(
        q, ikeys, ivals6, work_group_sorter5);
    std::cout << "KV joint sort (Key: uint8_t, Val: int32_t) <NUM =" << NUM
              << ", WG = 16> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int32_t, 32, NUM,
                                  decltype(work_group_sorter5)>(
        q, ikeys, ivals6, work_group_sorter5);
    std::cout << "KV joint sort (Key: uint8_t, Val: int32_t) <NUM =" << NUM
              << ", WG = 32> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint64_t, 1, NUM,
                                  decltype(work_group_sorter6)>(
        q, ikeys, ivals7, work_group_sorter6);
    std::cout << "KV joint sort (Key: uint8_t, Val: uint64_t) <NUM =" << NUM
              << ", WG = 1> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint64_t, 2, NUM,
                                  decltype(work_group_sorter6)>(
        q, ikeys, ivals7, work_group_sorter6);
    std::cout << "KV joint sort (Key: uint8_t, Val: uint64_t) <NUM =" << NUM
              << ", WG = 2> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint64_t, 4, NUM,
                                  decltype(work_group_sorter6)>(
        q, ikeys, ivals7, work_group_sorter6);
    std::cout << "KV joint sort (Key: uint8_t, Val: uint64_t) <NUM =" << NUM
              << ", WG = 4> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint64_t, 8, NUM,
                                  decltype(work_group_sorter6)>(
        q, ikeys, ivals7, work_group_sorter6);
    std::cout << "KV joint sort (Key: uint8_t, Val: uint64_t) <NUM =" << NUM
              << ", WG = 8> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint64_t, 16, NUM,
                                  decltype(work_group_sorter6)>(
        q, ikeys, ivals7, work_group_sorter6);
    std::cout << "KV joint sort (Key: uint8_t, Val: uint64_t) <NUM =" << NUM
              << ", WG = 16> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, uint64_t, 32, NUM,
                                  decltype(work_group_sorter6)>(
        q, ikeys, ivals7, work_group_sorter6);
    std::cout << "KV joint sort (Key: uint8_t, Val: uint64_t) <NUM =" << NUM
              << ", WG = 32> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int64_t, 1, NUM,
                                  decltype(work_group_sorter7)>(
        q, ikeys, ivals8, work_group_sorter7);
    std::cout << "KV joint sort (Key: uint8_t, Val: int64_t) <NUM =" << NUM
              << ", WG = 1> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int64_t, 2, NUM,
                                  decltype(work_group_sorter7)>(
        q, ikeys, ivals8, work_group_sorter7);
    std::cout << "KV joint sort (Key: uint8_t, Val: int64_t) <NUM =" << NUM
              << ", WG = 2> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int64_t, 4, NUM,
                                  decltype(work_group_sorter7)>(
        q, ikeys, ivals8, work_group_sorter7);
    std::cout << "KV joint sort (Key: uint8_t, Val: int64_t) <NUM =" << NUM
              << ", WG = 4> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int64_t, 8, NUM,
                                  decltype(work_group_sorter7)>(
        q, ikeys, ivals8, work_group_sorter7);
    std::cout << "KV joint sort (Key: uint8_t, Val: int64_t) <NUM =" << NUM
              << ", WG = 8> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int64_t, 16, NUM,
                                  decltype(work_group_sorter7)>(
        q, ikeys, ivals8, work_group_sorter7);
    std::cout << "KV joint sort (Key: uint8_t, Val: int64_t) <NUM =" << NUM
              << ", WG = 16> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, int64_t, 32, NUM,
                                  decltype(work_group_sorter7)>(
        q, ikeys, ivals8, work_group_sorter7);
    std::cout << "KV joint sort (Key: uint8_t, Val: int64_t) <NUM =" << NUM
              << ", WG = 32> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, float, 1, NUM,
                                  decltype(work_group_sorter8)>(
        q, ikeys, ivals9, work_group_sorter8);
    std::cout << "KV joint sort (Key: uint8_t, Val: float) <NUM =" << NUM
              << ", WG = 1> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, float, 2, NUM,
                                  decltype(work_group_sorter8)>(
        q, ikeys, ivals9, work_group_sorter8);
    std::cout << "KV joint sort (Key: uint8_t, Val: float) <NUM =" << NUM
              << ", WG = 2> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, float, 4, NUM,
                                  decltype(work_group_sorter8)>(
        q, ikeys, ivals9, work_group_sorter8);
    std::cout << "KV joint sort (Key: uint8_t, Val: float) <NUM =" << NUM
              << ", WG = 4> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, float, 8, NUM,
                                  decltype(work_group_sorter8)>(
        q, ikeys, ivals9, work_group_sorter8);
    std::cout << "KV joint sort (Key: uint8_t, Val: float) <NUM =" << NUM
              << ", WG = 8> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, float, 16, NUM,
                                  decltype(work_group_sorter8)>(
        q, ikeys, ivals9, work_group_sorter8);
    std::cout << "KV joint sort (Key: uint8_t, Val: float) <NUM =" << NUM
              << ", WG = 16> pass." << std::endl;

    test_work_group_KV_joint_sort<uint8_t, float, 32, NUM,
                                  decltype(work_group_sorter8)>(
        q, ikeys, ivals9, work_group_sorter8);
    std::cout << "KV joint sort (Key: uint8_t, Val: float) <NUM =" << NUM
              << ", WG = 32> pass." << std::endl;
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
#if __DEVICE_CODE
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
    std::cout << "KV joint sort (Key: uint32_t, Val: uint32_t) <NUM = 21, WG = "
                 "16> pass."
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

#if __DEVICE_CODE
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
    std::cout << "KV joint sort (Key: uint32_t, Val: uint32_t) <NUM = " << NUM
              << ", WG = "
                 "2> pass."
              << std::endl;

    test_work_group_KV_joint_sort<uint32_t, uint32_t, 4, NUM,
                                  decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout << "KV joint sort (Key: uint32_t, Val: uint32_t) <NUM = " << NUM
              << ", WG = "
                 "4> pass."
              << std::endl;

    test_work_group_KV_joint_sort<uint32_t, uint32_t, 8, NUM,
                                  decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout << "KV joint sort (Key: uint32_t, Val: uint32_t) <NUM = " << NUM
              << ", WG = "
                 "8> pass."
              << std::endl;

    test_work_group_KV_joint_sort<uint32_t, uint32_t, 16, NUM,
                                  decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout << "KV joint sort (Key: uint32_t, Val: uint32_t) <NUM = " << NUM
              << ", WG = "
                 "16> pass."
              << std::endl;

    test_work_group_KV_joint_sort<uint32_t, uint32_t, 32, NUM,
                                  decltype(work_group_sorter)>(
        q, ikeys, ivals, work_group_sorter);
    std::cout << "KV joint sort (Key: uint32_t, Val: uint32_t) <NUM = " << NUM
              << ", WG = "
                 "32> pass."
              << std::endl;
  }

  return 0;
}
