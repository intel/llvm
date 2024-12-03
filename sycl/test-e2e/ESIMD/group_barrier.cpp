//==----- group_barrier.cpp - ESIMD root group barrier test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc || gpu-intel-dg2
// REQUIRES-INTEL-DRIVER: lin: 31155

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "esimd_test_utils.hpp"
#include <sycl/ext/oneapi/experimental/root_group.hpp>
#include <sycl/group_barrier.hpp>

static constexpr int WorkGroupSize = 16;

static constexpr int VL = 16;
template <bool UseThisWorkItemAPI> bool test(sycl::queue &q) {
  bool Pass = true;
  const auto MaxWGs = 8;
  size_t WorkItemCount = MaxWGs * WorkGroupSize * VL;
  std::cout << "Test case UseThisWorkItemAPI="
            << std::to_string(UseThisWorkItemAPI) << std::endl;
  const auto Props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::use_root_sync};
  sycl::buffer<int> DataBuf{sycl::range{WorkItemCount}};
  const auto Range = sycl::nd_range<1>{MaxWGs * WorkGroupSize, WorkGroupSize};
  q.submit([&](sycl::handler &h) {
     sycl::accessor Data{DataBuf, h};
     h.parallel_for(Range, Props, [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
       int ID = it.get_global_linear_id();
       __ESIMD_NS::simd<int, VL> V(ID, 1);
       // Write data to another kernel's data to verify the barrier works.
       __ESIMD_NS::block_store(
           Data, (WorkItemCount * sizeof(int)) - (ID * sizeof(int) * VL), V);
       if constexpr (UseThisWorkItemAPI) {
         auto Root =
             sycl::ext::oneapi::experimental::this_work_item::get_root_group<
                 1>();
         sycl::group_barrier(Root);
       } else {
         auto Root = it.ext_oneapi_get_root_group();
         sycl::group_barrier(Root);
       }
       __ESIMD_NS::simd<int, VL> VOther(ID * VL, 1);
       __ESIMD_NS::block_store(Data, ID * sizeof(int) * VL, VOther);
     });
   }).wait();
  sycl::host_accessor Data{DataBuf};
  int ErrCnt = 0;
  for (int I = 0; I < WorkItemCount; I++) {
    if (Data[I] != I) {
      Pass = false;
      if (++ErrCnt < 16)
        std::cout << "Data[" << std::to_string(I)
                  << "] != " << std::to_string(I) << "\n";
    }
  }
  return Pass;
}
int main() {
  sycl::queue q;
  esimd_test::printTestLabel(q);
  bool Pass = true;
  Pass &= test<true>(q);
  Pass &= test<false>(q);
  if (Pass)
    std::cout << "Passed\n";
  else
    std::cout << "Failed\n";
  return !Pass;
}
