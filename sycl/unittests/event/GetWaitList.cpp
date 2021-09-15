//==--------------------- GetWaitList.cpp --- event unit test --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <gtest/gtest.h>

TEST(GetWaitList, GetWaitListTest) {
  sycl::queue q = sycl::queue(sycl::default_selector());

  sycl::event eA = q.submit([&](sycl::handler &cgh) {
      cgh.host_task([]() {
        double p = 0;
        p++; 
      });
  });
  sycl::event eB = q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(eA);
      cgh.host_task([]() {
        double p = 0;
        p++; 
      });
  });
  sycl::event eC = q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(eA);
      cgh.host_task([]() {
        double p = 0;
        p++; 
      });
  });

  auto res = eC.get_wait_list();
  assert(res.size() == 1);
  ASSERT_EQ(res[0], eA);

  sycl::event eD = q.submit([&](sycl::handler &cgh) {
      cgh.depends_on({eB,eC});
      cgh.host_task([]() {
        double p = 0;
        p++; 
      });
  });

  res = eD.get_wait_list();
  assert(res.size() == 2);
  ASSERT_EQ(res[0], eB);
  ASSERT_EQ(res[1], eC);

  eD.wait();

}
