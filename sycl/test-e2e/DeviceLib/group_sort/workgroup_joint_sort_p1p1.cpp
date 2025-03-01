// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// RUN: %{build}  -fsycl-device-lib-jit-link -o %t.out
// RUN: %{run} %t.out

// RUN: %{build} -DDES -o %t.out
// RUN: %{run} %t.out

// RUN: %{build} -DDES  -fsycl-device-lib-jit-link -o %t.out
// RUN: %{run} %t.out
//
// UNSUPPORTED: cuda || hip

#include "group_joint_sort_p1p1.hpp"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

template <typename Ty, size_t WG_SZ, size_t NUM, typename SortHelper>
void test_work_group_joint_sort(sycl::queue &q, Ty input[NUM], SortHelper gsh) {
  Ty scratch[NUM] = {
      0,
  };
  Ty result[NUM];
  memcpy(result, input, sizeof(Ty) * NUM);
  std::sort(&result[0], &result[NUM]);
  const static size_t wg_size = WG_SZ;
  nd_range<1> num_items((range<1>(wg_size)), (range<1>(wg_size)));
  {
    buffer<Ty, 1> ibuf(input, NUM);
    buffer<Ty, 1> obuf(scratch, NUM);
    q.submit([&](auto &h) {
       accessor in_acc{ibuf, h};
       accessor out_acc{obuf, h};
       h.parallel_for(num_items, [=](nd_item<1> i) {
         Ty *optr =
             out_acc.template get_multi_ptr<access::decorated::no>().get();
         uint8_t *by = reinterpret_cast<uint8_t *>(optr);
         gsh(in_acc.template get_multi_ptr<access::decorated::no>().get(), NUM,
             by);
       });
     }).wait();
  }

  bool fails = false;
  for (size_t idx = 0; idx < NUM; ++idx) {
#ifdef DES
    if (input[idx] != result[NUM - 1 - idx]) {
#else
    if (input[idx] != result[idx]) {
#endif
      fails = true;
      break;
    }
  }
  assert(!fails);
}

int main() {
  queue q;

  {
    constexpr static int NUM = 19;
    int8_t a[NUM] = {-1,  11, 10, 9, 3,   100, 34, 8,  10, 77,
                     -93, 23, 36, 2, 111, 91,  88, -2, -25};
    auto work_group_sorter = [](int8_t *first, uint32_t n, uint8_t *scratch) {
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1i8_u32_p1i8(
          first, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1i8_u32_p1i8(
          first, n, scratch);
#endif
    };
    test_work_group_joint_sort<int8_t, 8, NUM, decltype(work_group_sorter)>(
        q, a, work_group_sorter);
    std::cout << "work group joint sort p1i8_u32_p1i8 passes" << std::endl;
  }

  {
    constexpr static int NUM = 19;
    int32_t a[NUM] = {-1,  11, 1,  9, 3,   100, 34, 8,  1000, 77,
                      293, 23, 36, 2, 111, 91,  88, -2, 525};
    auto work_group_sorter = [](int32_t *first, uint32_t n, uint8_t *scratch) {
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1i32_u32_p1i8(
          first, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1i32_u32_p1i8(
          first, n, scratch);
#endif
    };
    test_work_group_joint_sort<int32_t, 8, NUM, decltype(work_group_sorter)>(
        q, a, work_group_sorter);
    std::cout << "work group joint sort p1i32_u32_p1i8 passes" << std::endl;
  }

  {
    constexpr static int NUM = 21;
    int16_t a[NUM] = {-1, 11, 1, 9,   3,  100, 34, 8,   1000, 77, 293,
                      23, 36, 2, 111, 91, 88,  -2, 525, -12,  525};
    auto work_group_sorter = [](int16_t *first, uint32_t n, uint8_t *scratch) {
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1i16_u32_p1i8(
          first, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1i16_u32_p1i8(
          first, n, scratch);
#endif
    };
    test_work_group_joint_sort<int16_t, 8, NUM, decltype(work_group_sorter)>(
        q, a, work_group_sorter);
    std::cout << "work group joint sort p1i16_u32_p1i8 passes" << std::endl;
  }

  {
    constexpr static int NUM = 23;
    int64_t a[NUM] = {-1,   11, 1,   9,   3,   100,       34,         8,
                      1000, 77, 293, 23,  36,  2,         111,        91,
                      88,   -2, 525, -12, 525, -99999999, 19928348493};
    auto work_group_sorter = [](int64_t *first, uint32_t n, uint8_t *scratch) {
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1i64_u32_p1i8(
          first, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1i64_u32_p1i8(
          first, n, scratch);
#endif
    };
    test_work_group_joint_sort<int64_t, 8, NUM, decltype(work_group_sorter)>(
        q, a, work_group_sorter);
    std::cout << "work group joint sort p1i64_u32_p1i8 passes" << std::endl;
  }

  {
    constexpr static int NUM = 23;
    float a[NUM] = {-1.25f,  11.4643f,    1.45f,           -9.98f,   13.665f,
                    100.0f,  34.625f,     8.125f,          1000.12f, 77.91f,
                    293.33f, 23.4f,       -36.6f,          2.5f,     111.11f,
                    91.889f, 88.88f,      -2.98f,          525.25f,  -12.11f,
                    525.0f,  -9999999.9f, 19928348493.123f};
    auto work_group_sorter = [](float *first, uint32_t n, uint8_t *scratch) {
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1f32_u32_p1i8(
          first, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1f32_u32_p1i8(
          first, n, scratch);
#endif
    };
    test_work_group_joint_sort<float, 8, NUM, decltype(work_group_sorter)>(
        q, a, work_group_sorter);
    std::cout << "work group joint sort p1f32_u32_p1i8 passes" << std::endl;
  }

  {
    constexpr static int NUM = 23;
    uint8_t a[NUM] = {234, 11, 1,   9,  3,   100, 34, 8, 121, 77, 125, 23,
                      36,  2,  111, 91, 201, 211, 77, 8, 88,  19, 0};
    auto work_group_sorter = [](uint8_t *first, uint32_t n, uint8_t *scratch) {
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1u8_u32_p1i8(
          first, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1u8_u32_p1i8(
          first, n, scratch);
#endif
    };
    test_work_group_joint_sort<uint8_t, 8, NUM, decltype(work_group_sorter)>(
        q, a, work_group_sorter);
    std::cout << "work group joint sort p1u8_u32_p1i8 passes" << std::endl;
  }

  {
    constexpr static int NUM = 23;
    uint16_t a[NUM] = {11234, 11,  1,   119, 3,     100, 341, 8,
                       121,   77,  125, 23,  3226,  2,   111, 911,
                       201,   211, 77,  8,   11188, 19,  0};
    auto work_group_sorter = [](uint16_t *first, uint32_t n, uint8_t *scratch) {
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1u16_u32_p1i8(
          first, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1u16_u32_p1i8(
          first, n, scratch);
#endif
    };
    test_work_group_joint_sort<uint16_t, 8, NUM, decltype(work_group_sorter)>(
        q, a, work_group_sorter);
    std::cout << "work group joint sort p1u16_u32_p1i8 passes" << std::endl;
  }

  {
    constexpr static int NUM = 23;
    uint32_t a[NUM] = {11234,   11,  1,      1193332332, 231,   100, 341, 8,
                       121,     77,  125,    32,         3226,  2,   111, 911,
                       9912201, 211, 711117, 8,          11188, 19,  0};
    auto work_group_sorter = [](uint32_t *first, uint32_t n, uint8_t *scratch) {
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1u32_u32_p1i8(
          first, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1u32_u32_p1i8(
          first, n, scratch);
#endif
    };
    test_work_group_joint_sort<uint32_t, 8, NUM, decltype(work_group_sorter)>(
        q, a, work_group_sorter);
    std::cout << "work group joint sort p1u32_u32_p1i8 passes" << std::endl;
  }

  {
    constexpr static int NUM = 23;
    uint64_t a[NUM] = {0x112A111111FFEEFF,
                       0xAACC11,
                       0x1,
                       0x1193332332,
                       0x231,
                       0xAA,
                       0xFCCCA341,
                       0x8,
                       0x121,
                       0x987777777,
                       0x81,
                       0x20,
                       0x3226,
                       0x2,
                       0x8FFFFFFFFF111,
                       0x911,
                       0xAAAA9912201,
                       0x211,
                       0x711117,
                       0x8,
                       0xABABABABCC,
                       0x13,
                       0};
    auto work_group_sorter = [](uint64_t *first, uint32_t n, uint8_t *scratch) {
#ifdef DES
      __devicelib_default_work_group_joint_sort_descending_p1u64_u32_p1i8(
          first, n, scratch);
#else
      __devicelib_default_work_group_joint_sort_ascending_p1u64_u32_p1i8(
          first, n, scratch);
#endif
    };
    test_work_group_joint_sort<uint64_t, 8, NUM, decltype(work_group_sorter)>(
        q, a, work_group_sorter);
    std::cout << "work group joint sort p1u64_u32_p1i8 passes" << std::endl;
  }

  return 0;
}
