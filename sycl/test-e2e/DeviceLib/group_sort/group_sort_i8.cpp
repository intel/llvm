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

#include <cmath>
#include <cstddef>
#include <iostream>
#include <sycl.hpp>
using namespace sycl;
#ifdef __SYCL_DEVICE_ONLY__
SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1i8_u32_p3i8(
    int8_t *first, uint32_t n, uint8_t *scratch);

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1i8_u32_p3i8(
    int8_t *first, uint32_t n, uint8_t *scratch);
#else
extern "C" void
__devicelib_default_work_group_joint_sort_ascending_p1i8_u32_p3i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  return;
}

SYCL_EXTERNAL extern "C" void
__devicelib_default_work_group_joint_sort_descending_p1i8_u32_p3i8(
    int8_t *first, uint32_t n, uint8_t *scratch) {
  return;
}
#endif

constexpr static size_t NUM = 18;
int main() {
  queue q;
  int8_t a[NUM] = {-1, 11,  1,  9,  3, 100,  34, 8,   121,
                   77, 125, 23, 36, 2, -111, 91, -88, -2};
  int8_t b[NUM] = {
      0,
  };
  int8_t c[NUM];
  memcpy(c, a, sizeof(a));
  std::sort(&c[0], &c[NUM]);

  nd_range<1> num_items(range<1>(8), range<1>(8));
  {
    buffer<int8_t, 1> ibuf(a, NUM);
    buffer<int8_t, 1> obuf(b, NUM);
    q.submit([&](auto &h) {
       accessor in_acc{ibuf, h};
       accessor out_acc{obuf, h};
       h.parallel_for(num_items, [=](nd_item<1> i) {
         int8_t *optr =
             out_acc.template get_multi_ptr<access::decorated::no>().get();
         uint8_t *by = reinterpret_cast<uint8_t *>(optr);
#ifdef DES
         __devicelib_default_work_group_joint_sort_descending_p1i8_u32_p3i8(
             in_acc.template get_multi_ptr<access::decorated::no>().get(), NUM,
             by);
#else
         __devicelib_default_work_group_joint_sort_ascending_p1i8_u32_p3i8(
             in_acc.template get_multi_ptr<access::decorated::no>().get(), NUM, by);
#endif
       });
     }).wait();
  }

  bool fails = false;
  for (size_t idx = 0; idx < NUM; ++idx) {
#ifdef DES
    if (a[idx] != c[NUM - 1 - idx]) {
#else
    if (a[idx] != c[idx]) {
#endif
      fails = true;
      break;
    }
  }
  assert(!fails);
  std::cout << "Pass!" << std::endl;
  return 0;
}
