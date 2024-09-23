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

  /*for (size_t idx = 0; idx < NUM; ++idx) {
    std::cout << "key: " << (int)std::get<0>(sorted_vec[idx]) << " val: " <<
  (int)std::get<1>(sorted_vec[idx]) << std::endl;
  }*/

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

  /* for (size_t idx = 0; idx < NUM; ++idx) {
    std::cout << "key: " << (int)(input_keys[idx]) << " val: " <<
  (int)(input_vals[idx]) << std::endl;
  }*/

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
#if __DEVICE_CODE
#ifdef DES
      __devicelib_default_work_group_private_sort_close_descending_p1u32_p1u32_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_ascending_p1u32_p1u32_u32_p1i8(
          keys, vals, n, scratch);
#endif
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
    uint8_t ivals1[NUM] = {99, 32,  1,   2,  67, 91,  45, 43,  91,  77,  16, 14,
                           24, 88,  76,  96, 76, 100, 63, 90,  52,  82,  1,  22,
                           9,  225, 127, 0,  12, 128, 3,  102, 200, 111, 123};
    int8_t ivals2[NUM] = {-99, 127, -121, 100, 9,  5,  12,  35,  -98,
                          77,  112, -91,  11,  12, 3,  71,  -66, 121,
                          18,  14,  21,   -22, 54, 88, -81, 31,  23,
                          53,  97,  103,  71,  83, 97, 37,  -41};
    uint16_t ivals3[NUM] = {28831, 23870, 54250, 5022,  9571,  60147, 9554,
                            18818, 28689, 18229, 40512, 23200, 40454, 24841,
                            43251, 63264, 29448, 45917, 882,   30788, 7586,
                            57541, 22108, 59535, 31880, 7152,  63919, 58703,
                            14686, 29914, 5872,  35868, 51479, 22721, 50927};
    int16_t ivals4[NUM] = {
        2798,  -13656, 1592,   3992,  -25870, 25172,  7761,   -18347, 1617,
        25472, 26763,  -5982,  24791, 27189,  22911,  22502,  15801,  25326,
        -2196, 9205,   -10418, 20464, -16616, -11285, 7249,   22866,  30574,
        -1298, 31351,  28252,  21322, -10072, 7874,   -26785, 22016};

    uint32_t ivals5[NUM] = {
        2238578408, 102907035,  2316773768, 617902655,  532045482,  73173328,
        1862406505, 142735533,  3494078873, 610196959,  4210902254, 1863122236,
        1257721692, 30008197,   3199012044, 3503276708, 3504950001, 1240383071,
        2463430884, 904104390,  4044803029, 3164373711, 1586440767, 1999536602,
        3377662770, 927279985,  1740225703, 1133653675, 3975816601, 260339911,
        1115507520, 2279020820, 4289105012, 692964674,  53775301};

    int32_t ivals6[NUM] = {
        507394811,   1949685322, 1624859474, -940434061,  -1440675113,
        -2002743224, 369969519,  840772268,  224522238,   296113452,
        -714007528,  480713824,  665592454,  1696360848,  780843358,
        -1901994531, 1667711523, 1390737696, 1357434904,  -290165630,
        305128121,   1301489180, 630469211,  -1385846315, 809333959,
        1098974670,  56900257,   876775101,  -1496897817, 1172877939,
        1528916082,  559152364,  749878571,  2071902702,  -430851798};

    auto work_group_sorter = [](uint8_t *keys, uint32_t *vals, uint32_t n,
                                uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
      __devicelib_default_work_group_private_sort_close_descending_p1u8_p1u32_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_ascending_p1u8_p1u32_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    auto work_group_sorter1 = [](uint8_t *keys, uint8_t *vals, uint32_t n,
                                 uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
      __devicelib_default_work_group_private_sort_close_descending_p1u8_p1u8_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_ascending_p1u8_p1u8_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    auto work_group_sorter2 = [](uint8_t *keys, int8_t *vals, uint32_t n,
                                 uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
      __devicelib_default_work_group_private_sort_close_descending_p1u8_p1i8_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_ascending_p1u8_p1i8_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    auto work_group_sorter3 = [](uint8_t *keys, uint16_t *vals, uint32_t n,
                                 uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
      __devicelib_default_work_group_private_sort_close_descending_p1u8_p1u16_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_ascending_p1u8_p1u16_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    auto work_group_sorter4 = [](uint8_t *keys, int16_t *vals, uint32_t n,
                                 uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
      __devicelib_default_work_group_private_sort_close_descending_p1u8_p1i16_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_ascending_p1u8_p1i16_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    auto work_group_sorter5 = [](uint8_t *keys, uint32_t *vals, uint32_t n,
                                 uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
      __devicelib_default_work_group_private_sort_close_descending_p1u8_p1u32_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_ascending_p1u8_p1u32_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    auto work_group_sorter6 = [](uint8_t *keys, int32_t *vals, uint32_t n,
                                 uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
      __devicelib_default_work_group_private_sort_close_descending_p1u8_p1i32_u32_p1i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_ascending_p1u8_p1i32_u32_p1i8(
          keys, vals, n, scratch);
#endif
#endif
    };

    constexpr static int NUM1 = 32;
    test_work_group_KV_private_sort<uint8_t, uint8_t, 1, NUM1,
                                    decltype(work_group_sorter1)>(
        q, ikeys, ivals1, work_group_sorter1);
    std::cout << "KV private sort <Key: uint8_t, Val: uint8_t> NUM = " << NUM1
              << ", WG = 1 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, uint8_t, 2, NUM1,
                                    decltype(work_group_sorter1)>(
        q, ikeys, ivals1, work_group_sorter1);
    std::cout << "KV private sort <Key: uint8_t, Val: uint8_t> NUM = " << NUM1
              << ", WG = 2 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, uint8_t, 4, NUM1,
                                    decltype(work_group_sorter1)>(
        q, ikeys, ivals1, work_group_sorter1);
    std::cout << "KV private sort <Key: uint8_t, Val: uint8_t> NUM = " << NUM1
              << ", WG = 4 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, uint8_t, 8, NUM1,
                                    decltype(work_group_sorter1)>(
        q, ikeys, ivals1, work_group_sorter1);
    std::cout << "KV private sort <Key: uint8_t, Val: uint8_t> NUM = " << NUM1
              << ", WG = 8 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, uint8_t, 16, NUM1,
                                    decltype(work_group_sorter1)>(
        q, ikeys, ivals1, work_group_sorter1);
    std::cout << "KV private sort <Key: uint8_t, Val: uint8_t> NUM = " << NUM1
              << ", WG = 16 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, uint8_t, 32, NUM1,
                                    decltype(work_group_sorter1)>(
        q, ikeys, ivals1, work_group_sorter1);
    std::cout << "KV private sort <Key: uint8_t, Val: uint8_t> NUM = " << NUM1
              << ", WG = 32 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, int8_t, 1, NUM,
                                    decltype(work_group_sorter2)>(
        q, ikeys, ivals2, work_group_sorter2);
    std::cout << "KV private sort <Key: uint8_t, Val: int8_t> NUM = " << NUM
              << ", WG = 1 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, int8_t, 5, NUM,
                                    decltype(work_group_sorter2)>(
        q, ikeys, ivals2, work_group_sorter2);
    std::cout << "KV private sort <Key: uint8_t, Val: int8_t> NUM = " << NUM
              << ", WG = 5 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, int8_t, 7, NUM,
                                    decltype(work_group_sorter2)>(
        q, ikeys, ivals2, work_group_sorter2);
    std::cout << "KV private sort <Key: uint8_t, Val: int8_t> NUM = " << NUM
              << ", WG = 7 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, int8_t, 35, NUM,
                                    decltype(work_group_sorter2)>(
        q, ikeys, ivals2, work_group_sorter2);
    std::cout << "KV private sort <Key: uint8_t, Val: int8_t> NUM = " << NUM
              << ", WG = 35 pass." << std::endl;

    constexpr static int NUM2 = 24;
    test_work_group_KV_private_sort<uint8_t, uint16_t, 1, NUM2,
                                    decltype(work_group_sorter3)>(
        q, ikeys, ivals3, work_group_sorter3);
    std::cout << "KV private sort <Key: uint8_t, Val: uint16_t> NUM = " << NUM2
              << ", WG = 1 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, uint16_t, 2, NUM2,
                                    decltype(work_group_sorter3)>(
        q, ikeys, ivals3, work_group_sorter3);
    std::cout << "KV private sort <Key: uint8_t, Val: uint16_t> NUM = " << NUM2
              << ", WG = 2 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, uint16_t, 3, NUM2,
                                    decltype(work_group_sorter3)>(
        q, ikeys, ivals3, work_group_sorter3);
    std::cout << "KV private sort <Key: uint8_t, Val: uint16_t> NUM = " << NUM2
              << ", WG = 3 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, uint16_t, 4, NUM2,
                                    decltype(work_group_sorter3)>(
        q, ikeys, ivals3, work_group_sorter3);
    std::cout << "KV private sort <Key: uint8_t, Val: uint16_t> NUM = " << NUM2
              << ", WG = 4 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, uint16_t, 6, NUM2,
                                    decltype(work_group_sorter3)>(
        q, ikeys, ivals3, work_group_sorter3);
    std::cout << "KV private sort <Key: uint8_t, Val: uint16_t> NUM = " << NUM2
              << ", WG = 6 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, uint16_t, 8, NUM2,
                                    decltype(work_group_sorter3)>(
        q, ikeys, ivals3, work_group_sorter3);
    std::cout << "KV private sort <Key: uint8_t, Val: uint16_t> NUM = " << NUM2
              << ", WG = 8 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, uint16_t, 12, NUM2,
                                    decltype(work_group_sorter3)>(
        q, ikeys, ivals3, work_group_sorter3);
    std::cout << "KV private sort <Key: uint8_t, Val: uint16_t> NUM = " << NUM2
              << ", WG = 12 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, uint16_t, 24, NUM2,
                                    decltype(work_group_sorter3)>(
        q, ikeys, ivals3, work_group_sorter3);
    std::cout << "KV private sort <Key: uint8_t, Val: uint16_t> NUM = " << NUM2
              << ", WG = 24 pass." << std::endl;

    constexpr static int NUM3 = 20;
    test_work_group_KV_private_sort<uint8_t, int16_t, 1, NUM3,
                                    decltype(work_group_sorter4)>(
        q, ikeys, ivals4, work_group_sorter4);
    std::cout << "KV private sort <Key: uint8_t, Val: int16_t> NUM = " << NUM3
              << ", WG = 1 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, int16_t, 2, NUM3,
                                    decltype(work_group_sorter4)>(
        q, ikeys, ivals4, work_group_sorter4);
    std::cout << "KV private sort <Key: uint8_t, Val: int16_t> NUM = " << NUM3
              << ", WG = 2 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, int16_t, 4, NUM3,
                                    decltype(work_group_sorter4)>(
        q, ikeys, ivals4, work_group_sorter4);
    std::cout << "KV private sort <Key: uint8_t, Val: int16_t> NUM = " << NUM3
              << ", WG = 4 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, int16_t, 5, NUM3,
                                    decltype(work_group_sorter4)>(
        q, ikeys, ivals4, work_group_sorter4);
    std::cout << "KV private sort <Key: uint8_t, Val: int16_t> NUM = " << NUM3
              << ", WG = 5 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, int16_t, 10, NUM3,
                                    decltype(work_group_sorter4)>(
        q, ikeys, ivals4, work_group_sorter4);
    std::cout << "KV private sort <Key: uint8_t, Val: int16_t> NUM = " << NUM3
              << ", WG = 10 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, int16_t, 20, NUM3,
                                    decltype(work_group_sorter4)>(
        q, ikeys, ivals4, work_group_sorter4);
    std::cout << "KV private sort <Key: uint8_t, Val: int16_t> NUM = " << NUM3
              << ", WG = 20 pass." << std::endl;

    constexpr static int NUM4 = 30;
    test_work_group_KV_private_sort<uint8_t, uint32_t, 1, NUM4,
                                    decltype(work_group_sorter5)>(
        q, ikeys, ivals5, work_group_sorter5);
    std::cout << "KV private sort <Key: uint8_t, Val: uint32_t> NUM = " << NUM4
              << ", WG = 1 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, uint32_t, 2, NUM4,
                                    decltype(work_group_sorter5)>(
        q, ikeys, ivals5, work_group_sorter5);
    std::cout << "KV private sort <Key: uint8_t, Val: uint32_t> NUM = " << NUM4
              << ", WG = 2 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, uint32_t, 3, NUM4,
                                    decltype(work_group_sorter5)>(
        q, ikeys, ivals5, work_group_sorter5);
    std::cout << "KV private sort <Key: uint8_t, Val: uint16_t> NUM = " << NUM4
              << ", WG = 3 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, uint32_t, 5, NUM4,
                                    decltype(work_group_sorter5)>(
        q, ikeys, ivals5, work_group_sorter5);
    std::cout << "KV private sort <Key: uint8_t, Val: uint32_t> NUM = " << NUM4
              << ", WG = 5 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, uint32_t, 6, NUM4,
                                    decltype(work_group_sorter5)>(
        q, ikeys, ivals5, work_group_sorter5);
    std::cout << "KV private sort <Key: uint8_t, Val: uint32_t> NUM = " << NUM4
              << ", WG = 6 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, uint32_t, 10, NUM4,
                                    decltype(work_group_sorter5)>(
        q, ikeys, ivals5, work_group_sorter5);
    std::cout << "KV private sort <Key: uint8_t, Val: uint32_t> NUM = " << NUM4
              << ", WG = 10 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, uint32_t, 15, NUM4,
                                    decltype(work_group_sorter5)>(
        q, ikeys, ivals5, work_group_sorter5);
    std::cout << "KV private sort <Key: uint8_t, Val: uint32_t> NUM = " << NUM4
              << ", WG = 15 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, uint32_t, 30, NUM4,
                                    decltype(work_group_sorter5)>(
        q, ikeys, ivals5, work_group_sorter5);
    std::cout << "KV private sort <Key: uint8_t, Val: uint32_t> NUM = " << NUM4
              << ", WG = 30 pass." << std::endl;

    constexpr static int NUM5 = 25;
    test_work_group_KV_private_sort<uint8_t, int32_t, 1, NUM5,
                                    decltype(work_group_sorter6)>(
        q, ikeys, ivals6, work_group_sorter6);
    std::cout << "KV private sort <Key: uint8_t, Val: int32_t> NUM = " << NUM5
              << ", WG = 1 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, int32_t, 5, NUM5,
                                    decltype(work_group_sorter6)>(
        q, ikeys, ivals6, work_group_sorter6);
    std::cout << "KV private sort <Key: uint8_t, Val: int32_t> NUM = " << NUM5
              << ", WG = 5 pass." << std::endl;

    test_work_group_KV_private_sort<uint8_t, int32_t, 25, NUM5,
                                    decltype(work_group_sorter6)>(
        q, ikeys, ivals6, work_group_sorter6);
    std::cout << "KV private sort <Key: uint8_t, Val: int32_t> NUM = " << NUM5
              << ", WG = 25 pass." << std::endl;
  }
}
