// UNSUPPORTED: cuda
// CUDA compilation and runtime do not yet support sub-groups.
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

//==- free_function_queries_sub_group.cpp - SYCL free queries for sub group -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------===//

#include "../../sub_group/helper.hpp"
#include <CL/sycl.hpp>

#include <cassert>
#include <iostream>
#include <type_traits>

int main() {
  constexpr std::size_t n = 10;

  int data[n]{};
  int counter{0};
  {
    {
      sycl::buffer<int> buf(data, sycl::range<1>(n));
      sycl::queue q;
      if (!core_sg_supported(q.get_device())) {
        std::cout << "Skipping test" << std::endl;
        return 0;
      }
      sycl::nd_range<1> NDR(sycl::range<1>{n}, sycl::range<1>{2});
      q.submit([&](cl::sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access::mode::write,
                       sycl::access::target::global_buffer>
            acc(buf.get_access<sycl::access::mode::write>(cgh));
        cgh.parallel_for<class NdItemTest>(NDR, [=](auto nd_i) {
          static_assert(std::is_same<decltype(nd_i), sycl::nd_item<1>>::value,
                        "lambda arg type is unexpected");
          auto that_nd_item = sycl::this_nd_item<1>();
          assert(that_nd_item == nd_i);
          auto that_sub_group = sycl::ONEAPI::this_sub_group();
          assert(that_sub_group.get_local_linear_id() ==
                 that_nd_item.get_sub_group().get_local_linear_id());
          assert(that_sub_group.get_local_id() ==
                 that_nd_item.get_sub_group().get_local_id());
          assert(that_sub_group.get_local_range() ==
                 that_nd_item.get_sub_group().get_local_range());

          acc[that_nd_item.get_global_id(0)]++;
        });
      });
    }
    ++counter;
    for (int i = 0; i < n; i++) {
      assert(data[i] == counter);
    }
  }
}
