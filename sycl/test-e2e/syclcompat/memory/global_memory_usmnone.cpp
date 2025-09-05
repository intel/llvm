// ====------ global_memory.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
#define SYCLCOMPAT_USM_LEVEL_NONE
#include <sycl/detail/core.hpp>
#include <syclcompat/syclcompat.hpp>

class TestStruct {
public:
  void test() {}
  template<class T> void testTemplate() {}
};

template<class T>
class TemplateStuct {
public:
  void test() {}
  template<class Ty> void testTemplate() {}
};

syclcompat::global_memory<volatile int, 0> d1_a(0);
syclcompat::global_memory<int, 1> d2_a(36);
syclcompat::global_memory<TemplateStuct<int>, 0> d3_a;
syclcompat::global_memory<TestStruct, 0> d4_a;
syclcompat::constant_memory<int, 1> c1_a(16);
syclcompat::constant_memory<int, 0> c2_a;
syclcompat::constant_memory<TemplateStuct<int>, 0> c3_a;
syclcompat::constant_memory<TestStruct, 0> c4_a;

syclcompat::constant_memory<int, 2> c_2d_a(sycl::range<2>(5, 3),
{{0, 10, 20},
{30, 40, 50},
{60, 70, 80},
{90, 100, 110},
{120, 130, 140}});
syclcompat::constant_memory<int, 2> c_2d_b(sycl::range<2>(3, 5),
{{0, 10, 20, 30, 40},
{50, 60, 70, 80, 90},
{100, 110, 120, 130, 140}});
syclcompat::constant_memory<int, 2> c_2d_c(sycl::range<2>(3, 5),
                                     {0, 10, 20, 30, 40,
                                      50, 60, 70, 80, 90,
                                      100, 110, 120, 130, 140});
syclcompat::constant_memory<int, 3> c_3d(sycl::range<3>(2, 2, 4),
                                   {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                                    110, 120, 130, 140});
syclcompat::constant_memory<int, 1> c_1d(sycl::range<1>(15),
                                   {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                                    110, 120, 130, 140});

bool verify_init(int *data) {
  for(auto i = 0; i < 15; ++i) {
    if (data[i] != i * 10)
      return false;
  }
  return true;
}

bool verify() {
  const int size = 15;
  auto size_bytes = 15 * sizeof(int);

  int h_result[15];
  syclcompat::memcpy(h_result, c_2d_a.get_ptr(), size_bytes);
  if (!verify_init(h_result))
    return false;
  syclcompat::memcpy(h_result, c_2d_b.get_ptr(), size_bytes);
  if (!verify_init(h_result))
    return false;
  syclcompat::memcpy(h_result, c_2d_c.get_ptr(), size_bytes);
  if (!verify_init(h_result))
    return false;
  syclcompat::memcpy(h_result, c_3d.get_ptr(), size_bytes);
  if (!verify_init(h_result))
    return false;
  syclcompat::memcpy(h_result, c_1d.get_ptr(), size_bytes);
  if (!verify_init(h_result))
    return false;
  return true;
}

void test4(TemplateStuct<int> *d3, TestStruct *d4) {
  d3->test();
  d3->testTemplate<int>();
  d4->test();
  d4->testTemplate<int>();
}

void test3(TemplateStuct<int> c3, TestStruct c4) {
  c3.test();
  c3.testTemplate<int>();
  c4.test();
  c4.testTemplate<int>();
}

void test2(volatile int &a) {
  a = 3;
}

void test1(volatile int *acc_d1, int *acc_d2, int const *c1, int c2) {
  unsigned d_a = 1;
  *acc_d1 = 0;
  *acc_d2 = d_a;
  unsigned d_c = (unsigned)(*acc_d1);
  unsigned *d_d = (unsigned *)acc_d2;
  unsigned *d_e = (unsigned *)(acc_d2 + 5);
  int *d_f = acc_d2 - 6;
  test2(*acc_d1);
}

int main() {
  d1_a.init();
  d2_a.init();
  c1_a.init();
  c2_a.init();
  syclcompat::get_default_queue().submit(
    [&](sycl::handler &cgh) {
      auto d1_acc = d1_a.get_access(cgh);
      auto d2_acc = d2_a.get_access(cgh);
      auto c1_acc = c1_a.get_access(cgh);
      auto c2_acc = c2_a.get_access(cgh);
      cgh.parallel_for<syclcompat_kernel_name<class kernel_test1>>(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item) {
            test1(d1_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                  d2_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                  c1_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                  c2_acc);
          });
    });
  c3_a.init();
  c4_a.init();
  syclcompat::get_default_queue().submit(
    [&](sycl::handler &cgh) {
      auto c3_acc = c3_a.get_access(cgh);
      auto c4_acc = c4_a.get_access(cgh);
      cgh.parallel_for<syclcompat_kernel_name<class kernel_test2>>(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
        [=] (sycl::nd_item<3> item) {
          test3(c3_acc, c4_acc);
        });
    });

  sycl::queue *q = syclcompat::get_current_device().create_queue();
  d3_a.init(*q);
  d4_a.init(*q);
  q->submit(
    [&](sycl::handler &cgh) {
      auto d3_acc = d3_a.get_access(cgh);
      auto d4_acc = d4_a.get_access(cgh);
      cgh.parallel_for<syclcompat_kernel_name<class kernel_test3>>(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item) {
            test4(d3_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                  d4_acc.get_multi_ptr<sycl::access::decorated::no>().get());
          });
    });


  if (verify()) {
    printf("Init Constant Memory Success!\n");
    return 0;
  } else {
    printf("Init Constant Memory Fail!\n");
    return 1;
  }
}
