//==--------- broadcast.hpp - SYCL sub_group broadcast test ----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helper.hpp"
#include <sycl/sycl.hpp>
template <typename T> class sycl_subgr;
using namespace cl::sycl;
template <typename T> void check(queue &Queue) {
  const int G = 256, L = 64;
  try {
    nd_range<1> NdRange(G, L);
    buffer<T> syclbuf(G);
    buffer<size_t> sgsizebuf(1);
    Queue.submit([&](handler &cgh) {
      auto syclacc = syclbuf.template get_access<access::mode::read_write>(cgh);
      auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<sycl_subgr<T>>(NdRange, [=](nd_item<1> NdItem) {
        ext::oneapi::sub_group SG = NdItem.get_sub_group();
        /*Broadcast GID of element with SGLID == SGID % SGMLR*/
        syclacc[NdItem.get_global_id()] = ext::oneapi::broadcast(
            SG, T(NdItem.get_global_id(0)),
            SG.get_group_id() % SG.get_max_local_range()[0]);
        if (NdItem.get_global_id(0) == 0)
          sgsizeacc[0] = SG.get_max_local_range()[0];
      });
    });
    auto syclacc = syclbuf.template get_access<access::mode::read_write>();
    auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>();
    size_t sg_size = sgsizeacc[0];
    if (sg_size == 0)
      sg_size = L;
    int WGid = -1, SGid = 0;
    for (int j = 0; j < G; j++) {
      if (j % L % sg_size == 0) {
        SGid++;
      }
      if (j % L == 0) {
        WGid++;
        SGid = 0;
      }
      exit_if_not_equal<T>(syclacc[j],
                           L * WGid + SGid % sg_size + SGid * sg_size,
                           "broadcasted value");
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }
}
