//==-------- khr_static_addrspace_cast.cpp - static addrspace cast test ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS

#include <sycl/detail/core.hpp>
#include <sycl/khr/static_addrspace_cast.hpp>

int main() {

  sycl::queue Queue;

  sycl::range<1> NItems{1};

  sycl::buffer<int, 1> GlobalBuffer{NItems};
  sycl::buffer<bool, 1> ResultBuffer{NItems};

  Queue
      .submit([&](sycl::handler &cgh) {
        auto GlobalAccessor =
            GlobalBuffer.get_access<sycl::access::mode::read_write>(cgh);
        auto LocalAccessor = sycl::local_accessor<int>(1, cgh);
        auto ResultAccessor =
            ResultBuffer.get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for<class Kernel>(
            sycl::nd_range<1>(NItems, 1), [=](sycl::nd_item<1> Item) {
              bool Success = true;
              size_t Index = Item.get_global_id(0);

              int *RawGlobalPointer = &GlobalAccessor[Index];
              auto GlobalPointer = sycl::khr::static_addrspace_cast<
                  sycl::access::address_space::global_space>(RawGlobalPointer);
              Success &= reinterpret_cast<size_t>(RawGlobalPointer) ==
                         reinterpret_cast<size_t>(GlobalPointer.get_raw());

              int *RawLocalPointer = &LocalAccessor[0];
              auto LocalPointer = sycl::khr::static_addrspace_cast<
                  sycl::access::address_space::local_space>(RawLocalPointer);
              Success &= reinterpret_cast<size_t>(RawLocalPointer) ==
                         reinterpret_cast<size_t>(LocalPointer.get_raw());

              int PrivateVariable = 0;
              int *RawPrivatePointer = &PrivateVariable;
              auto PrivatePointer = sycl::khr::static_addrspace_cast<
                  sycl::access::address_space::private_space>(
                  RawPrivatePointer);
              Success &= reinterpret_cast<size_t>(RawPrivatePointer) ==
                         reinterpret_cast<size_t>(PrivatePointer.get_raw());

              ResultAccessor[Index] = Success;
            });
      })
      .wait();

  bool Success = true;
  {
    auto ResultAccessor = sycl::host_accessor(ResultBuffer);
    for (int i = 0; i < NItems.size(); ++i) {
      Success &= ResultAccessor[i];
    };
  }

  return (Success) ? 0 : -1;
}
