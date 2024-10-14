// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// RUN: %{build} -DCLOSE -o %t.out
// RUN: %{run} %t.out

// RUN: %{build}  -fsycl-device-lib-jit-link -o %t.out
// RUN: %{run} %t.out

// RUN: %{build}  -DCLOSE -fsycl-device-lib-jit-link -o %t.out
// RUN: %{run} %t.out

// RUN: %{build} -DDES -o %t.out
// RUN: %{run} %t.out

// RUN: %{build} -DDES -DCLOSE -o %t.out
// RUN: %{run} %t.out

// RUN: %{build} -DDES  -DCLOSE -fsycl-device-lib-jit-link -o %t.out
// RUN: %{run} %t.out
//
// UNSUPPORTED: cuda || hip

#include "group_private_sort_p1p1.hpp"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;
template <typename Ty, size_t WG_SZ, size_t NUM, typename SortHelper>
void test_work_group_private_sort(sycl::queue &q, Ty input[NUM],
                                  SortHelper gsh) {
  static_assert(NUM % WG_SZ == 0,
                "Input size must be divisible by Work group size!");
  // Scratch memory size >= NUM * sizeof(Ty) * 2
  Ty scratch[NUM * 2] = {
      0,
  };
  Ty result[NUM];
  Ty reference[NUM];
  memcpy(reference, input, sizeof(Ty) * NUM);
#ifdef DES
  std::sort(&reference[0], &reference[NUM], std::greater<Ty>());
#else
  std::sort(&reference[0], &reference[NUM]);
#endif
  const static size_t wg_size = WG_SZ;
  constexpr size_t num_per_work_item = NUM / WG_SZ;
  nd_range<1> num_items((range<1>(wg_size)), (range<1>(wg_size)));

  {
    buffer<Ty, 1> ibuf(input, NUM);
    buffer<Ty, 1> obuf(scratch, NUM * 2);
    buffer<Ty, 1> rbuf(result, NUM);
    q.submit([&](auto &h) {
       accessor in_acc{ibuf, h};
       accessor out_acc{obuf, h};
       accessor re_acc{rbuf, h};
       h.parallel_for(num_items, [=](nd_item<1> i) {
         Ty private_buf[num_per_work_item];
         for (size_t idx = 0; idx < num_per_work_item; ++idx)
           private_buf[idx] =
               in_acc[i.get_local_linear_id() * num_per_work_item + idx];
         group_barrier(i.get_group());
         Ty *optr =
             out_acc.template get_multi_ptr<access::decorated::no>().get();
         uint8_t *by = reinterpret_cast<uint8_t *>(optr);
         gsh(private_buf, num_per_work_item, by);
         for (size_t idx = 0; idx < num_per_work_item; ++idx)
           re_acc[i.get_local_linear_id() * num_per_work_item + idx] =
               private_buf[idx];
       });
     }).wait();
  }

  bool fails = false;

#ifdef CLOSE
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (result[idx] != reference[idx]) {
      fails = true;
      break;
    }
  }
#else
  for (size_t idx = 0; idx < NUM; ++idx) {
    size_t idx1 = idx % WG_SZ;
    size_t idx2 = idx / WG_SZ;
    if (reference[idx] != result[idx1 * num_per_work_item + idx2]) {
      fails = true;
      break;
    }
  }
#endif

  assert(!fails);
}

int main() {
  queue q;

  {
    constexpr static int NUM = 24;
    int8_t a[NUM] = {-1,  11,  10, 9,  3, 100, 34, 8,  10, 77,  10, 103,
                     -12, -93, 23, 36, 2, 111, 91, 88, -2, -25, 98, -111};
    auto work_group_sorter = [](int8_t *first, uint32_t n, uint8_t *scratch) {
#ifdef CLOSE
#ifdef DES
      __devicelib_default_work_group_private_sort_close_descending_p1i8_u32_p1i8(
          first, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_ascending_p1i8_u32_p1i8(
          first, n, scratch);
#endif
#else
#ifdef DES
      __devicelib_default_work_group_private_sort_spread_descending_p1i8_u32_p1i8(
          first, n, scratch);
#else
      __devicelib_default_work_group_private_sort_spread_ascending_p1i8_u32_p1i8(
          first, n, scratch);
#endif
#endif
    };
    test_work_group_private_sort<int8_t, 8, NUM, decltype(work_group_sorter)>(
        q, a, work_group_sorter);
    std::cout << "work group private sort p1i8_u32_p1i8 passes" << std::endl;
  }

  {
    constexpr static int NUM = 32;
    int16_t a[NUM] = {2162,   29891,  14709,  -20987, -30051, -26861, 5629,
                      -11244, 25702,  29438,  22560,  -15282, 27812,  28455,
                      26871,  -22327, 6495,   23519,  19389,  26328,  13253,
                      -24369, -1616,  3278,   5624,   -6317,  -3669,  11874,
                      -46,    -4717,  -27449, -9790};
    auto work_group_sorter = [](int16_t *first, uint32_t n, uint8_t *scratch) {
#ifdef CLOSE
#ifdef DES
      __devicelib_default_work_group_private_sort_close_descending_p1i16_u32_p1i8(
          first, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_ascending_p1i16_u32_p1i8(
          first, n, scratch);
#endif
#else
#ifdef DES
      __devicelib_default_work_group_private_sort_spread_descending_p1i16_u32_p1i8(
          first, n, scratch);
#else
      __devicelib_default_work_group_private_sort_spread_ascending_p1i16_u32_p1i8(
          first, n, scratch);
#endif
#endif
    };
    test_work_group_private_sort<int16_t, 8, NUM, decltype(work_group_sorter)>(
        q, a, work_group_sorter);
    std::cout << "work group private sort p1i16_u32_p1i8 passes" << std::endl;
  }

  {
    constexpr static int NUM = 32;
    int32_t a[NUM] = {1319329913,  -390041276,  -2040725419, -217333100,
                      -900793956,  -2138508211, 769705434,   122767310,
                      -1918605668, -16813517,   1616926513,  -2141526068,
                      631985359,   541606467,   662050754,   140359040,
                      1834119354,  1910851165,  809736505,   451506849,
                      -1713169862, -1916401837, 1490159094,  -2066441094,
                      -332318833,  -1550930943, 1763101596,  500568854,
                      -1574546569, -596440302,  1522396193,  -980468122};
    auto work_group_sorter = [](int32_t *first, uint32_t n, uint8_t *scratch) {
#ifdef CLOSE
#ifdef DES
      __devicelib_default_work_group_private_sort_close_descending_p1i32_u32_p1i8(
          first, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_ascending_p1i32_u32_p1i8(
          first, n, scratch);
#endif
#else
#ifdef DES
      __devicelib_default_work_group_private_sort_spread_descending_p1i32_u32_p1i8(
          first, n, scratch);
#else
      __devicelib_default_work_group_private_sort_spread_ascending_p1i32_u32_p1i8(
          first, n, scratch);
#endif
#endif
    };
    test_work_group_private_sort<int32_t, 8, NUM, decltype(work_group_sorter)>(
        q, a, work_group_sorter);
    std::cout << "work group private sort p1i32_u32_p1i8 passes" << std::endl;
  }
}
