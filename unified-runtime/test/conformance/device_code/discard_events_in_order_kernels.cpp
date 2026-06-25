// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Device code for UR reproducer of SYCL e2e test:
// sycl/test-e2e/DeprecatedFeatures/DiscardEvents/discard_events_l0_inorder.cpp

#include <sycl/sycl.hpp>

static constexpr int MAGIC_NUM1 = 2;

class DiscardEventsK1;
class DiscardEventsK2;
class DiscardEventsK3;
class DiscardEventsK4;
class DiscardEventsK5;

int main() {
  constexpr size_t buffer_size = 100;
  sycl::queue q;
  int *values1 = sycl::malloc_shared<int>(buffer_size, q);
  int *values2 = sycl::malloc_shared<int>(buffer_size, q);
  int *values3 = sycl::malloc_shared<int>(buffer_size, q);

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<DiscardEventsK1>(sycl::range<1>(buffer_size),
                                      [=](sycl::item<1> itemID) {
                                        size_t i = itemID.get_id(0);
                                        if (values1[i] == 0)
                                          if (values2[i] == static_cast<int>(i))
                                            if (values3[i] == static_cast<int>(i)) {
                                              values1[i] += static_cast<int>(i);
                                              values2[i] = MAGIC_NUM1;
                                              values3[i] = static_cast<int>(i);
                                            }
                                      });
  });

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<DiscardEventsK2>(sycl::range<1>(buffer_size),
                                      [=](sycl::item<1> itemID) {
                                        size_t i = itemID.get_id(0);
                                        if (values1[i] == static_cast<int>(i))
                                          if (values2[i] == MAGIC_NUM1)
                                            if (values3[i] == static_cast<int>(i))
                                              values1[i] += 10;
                                      });
  });

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<DiscardEventsK3>(sycl::range<1>(buffer_size),
                                      [=](sycl::item<1> itemID) {
                                        size_t i = itemID.get_id(0);
                                        if (values1[i] == static_cast<int>(i) + 10)
                                          if (values2[i] == static_cast<int>(i) + 10)
                                            if (values3[i] == static_cast<int>(i)) {
                                              values1[i] += 100;
                                              values2[i] = static_cast<int>(i);
                                            }
                                      });
  });

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<DiscardEventsK4>(sycl::range<1>(buffer_size),
                                      [=](sycl::item<1> itemID) {
                                        size_t i = itemID.get_id(0);
                                        if (values1[i] == static_cast<int>(i) + 110)
                                          if (values2[i] == static_cast<int>(i))
                                            if (values3[i] == static_cast<int>(i))
                                              values1[i] += 1000;
                                      });
  });

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<DiscardEventsK5>(sycl::range<1>(buffer_size),
                                      [=](sycl::item<1> itemID) {
                                        size_t i = itemID.get_id(0);
                                        if (values1[i] == static_cast<int>(i) + 1110)
                                          if (values2[i] == static_cast<int>(i))
                                            if (values3[i] == static_cast<int>(i))
                                              values1[i] += 10000;
                                      });
  });

  sycl::free(values1, q);
  sycl::free(values2, q);
  sycl::free(values3, q);
  return 0;
}