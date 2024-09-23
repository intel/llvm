//==------- shift_left_right.hpp -*- C++ -*---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helpers.hpp"
#include <sycl/detail/core.hpp>
#include <sycl/group_algorithm.hpp>
template <typename T, int N> class sycl_subgr;

using namespace sycl;

// TODO remove this workaround when clang will support correct generation of
// half typename in integration header
struct wa_half;

// ---- check
template <typename T, int N>
void check(queue &Queue, size_t G = 256, size_t L = 64) {
  try {
    nd_range<1> NdRange(G, L);
    buffer<vec<T, N>> buf_right(G);
    buffer<vec<T, N>> buf_left(G);
    buffer<size_t> sgsizebuf(1);
    Queue.submit([&](handler &cgh) {
      accessor acc_right{buf_right, cgh, sycl::read_write};
      accessor acc_left{buf_left, cgh, sycl::read_write};
      accessor sgsizeacc{sgsizebuf, cgh, sycl::read_write};

      cgh.parallel_for<sycl_subgr<T, N>>(NdRange, [=](nd_item<1> NdItem) {
        sycl::sub_group SG = NdItem.get_sub_group();
        uint32_t wggid = NdItem.get_global_id(0);
        uint32_t sgid = SG.get_group_id().get(0);
        vec<T, N> vwggid(wggid), vsgid(sgid);
        if (wggid == 0)
          sgsizeacc[0] = SG.get_max_local_range()[0];

        /* Save GID-SGID */
        acc_right[NdItem.get_global_id()] = shift_group_right(SG, vwggid, sgid);
        /* Save GID+SGID */
        acc_left[NdItem.get_global_id()] = shift_group_left(SG, vwggid, sgid);
      });
    });
    host_accessor acc_right{buf_right, sycl::read_write};
    host_accessor acc_left{buf_left, sycl::read_write};
    host_accessor sgsizeacc{sgsizebuf, sycl::read_write};

    size_t sg_size = sgsizeacc[0];
    int SGid = 0;
    int SGLid = 0;
    int SGBeginGid = 0;
    for (int j = 0; j < G; j++) {
      if (j % L % sg_size == 0) {
        SGid++;
        SGLid = 0;
        SGBeginGid = j;
      }
      if (j % L == 0) {
        SGid = 0;
        SGLid = 0;
        SGBeginGid = j;
      }

      /* Value GID+SGID for all element except last SGID in SG*/
      if (j % L % sg_size + SGid < sg_size && j % L + SGid < L) {
        exit_if_not_equal_vec(acc_left[j], vec<T, N>(j + SGid % sg_size),
                              "shift_group_left");
      }
      /* Value GID-SGID for all element except first SGID in SG*/
      if (j % L % sg_size >= SGid) {
        exit_if_not_equal_vec(acc_right[j], vec<T, N>(j - SGid % sg_size),
                              "shift_group_right");
      }

      SGLid++;
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }
}

template <typename T> void check(queue &Queue, size_t G = 256, size_t L = 64) {
  try {
    nd_range<1> NdRange(G, L);
    buffer<T> buf_right(G);
    buffer<T> buf_left(G);
    buffer<size_t> sgsizebuf(1);
    Queue.submit([&](handler &cgh) {
      accessor acc_right{buf_right, cgh, sycl::read_write};
      accessor acc_left{buf_left, cgh, sycl::read_write};
      accessor sgsizeacc{sgsizebuf, cgh, sycl::read_write};
      cgh.parallel_for<sycl_subgr<T, 0>>(NdRange, [=](nd_item<1> NdItem) {
        sycl::sub_group SG = NdItem.get_sub_group();
        uint32_t wggid = NdItem.get_global_id(0);
        uint32_t sgid = SG.get_group_id().get(0);
        if (wggid == 0)
          sgsizeacc[0] = SG.get_max_local_range()[0];

        /* Save GID-SGID */
        acc_right[NdItem.get_global_id()] = shift_group_right(SG, wggid, sgid);
        /* Save GID+SGID */
        acc_left[NdItem.get_global_id()] = shift_group_left(SG, wggid, sgid);
      });
    });
    host_accessor acc_right{buf_right, sycl::read_write};
    host_accessor acc_left{buf_left, sycl::read_write};
    host_accessor sgsizeacc{sgsizebuf, sycl::read_write};

    size_t sg_size = sgsizeacc[0];
    int SGid = 0;
    int SGLid = 0;
    int SGBeginGid = 0;
    for (int j = 0; j < G; j++) {
      if (j % L % sg_size == 0) {
        SGid++;
        SGLid = 0;
        SGBeginGid = j;
      }
      if (j % L == 0) {
        SGid = 0;
        SGLid = 0;
        SGBeginGid = j;
      }

      /* Value GID+SGID for all element except last SGID in SG*/
      if (j % L % sg_size + SGid < sg_size && j % L + SGid < L) {
        exit_if_not_equal<T>(acc_left[j], j + SGid, "shift_group_left");
      }
      /* Value GID-SGID for all element except first SGID in SG*/
      if (j % L % sg_size >= SGid) {
        exit_if_not_equal<T>(acc_right[j], j - SGid, "shift_group_right");
      }

      SGLid++;
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }
}
