// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// RUN: %{build}  -fsycl-device-lib-jit-link -o %t.out
// RUN: %{run} %t.out

// RUN: %{build} -DDES -o %t.out
// RUN: %{run} %t.out

// RUN: %{build} -DES  -fsycl-device-lib-jit-link -o %t.out
// RUN: %{run} %t.out
//
// UNSUPPORTED: cuda || hip

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <sycl.hpp>
using namespace sycl;
#ifdef __SYCL_DEVICE_ONLY__
SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1i32_u32_p1i8(
    int32_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1i32_u32_p1i8(
    int32_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1i16_u32_p1i8(
    int16_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1i16_u32_p1i8(
    int16_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1i64_u32_p1i8(
    int64_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1i64_u32_p1i8(
    int64_t *first, uint32_t n, uint8_t *scratch);
#else
extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1i32_u32_p1i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1i32_u32_p1i8(
    int32_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1i16_u32_p1i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1i16_u32_p1i8(
    int16_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1i64_u32_p1i8(
    int64_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1i64_u32_p1i8(
    int64_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

#endif

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

  return 0;
}
