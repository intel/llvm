//==----- permute_select.hpp -*- C++ -*------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helpers.hpp"
#include <sycl/sycl.hpp>
template <typename T, int N> class sycl_subgr;

using namespace cl::sycl;

// TODO remove this workaround when clang will support correct generation of
// half typename in integration header
struct wa_half;

template <typename T, int N>
void check(queue &Queue, size_t G = 256, size_t L = 64) {
  try {
    nd_range<1> NdRange(G, L);
    buffer<vec<T, N>> buf_select(G);
    buffer<vec<T, N>> buf_xor(G);
    buffer<size_t> sgsizebuf(1);
    Queue.submit([&](handler &cgh) {
      accessor acc_select{buf_select, cgh, sycl::read_write};
      accessor acc_xor{buf_xor, cgh, sycl::read_write};
      accessor sgsizeacc{sgsizebuf, cgh, sycl::read_write};
      cgh.parallel_for<sycl_subgr<T, N>>(NdRange, [=](nd_item<1> NdItem) {
        ext::oneapi::sub_group SG = NdItem.get_sub_group();
        uint32_t wggid = NdItem.get_global_id(0);
        uint32_t sgid = SG.get_group_id().get(0);
        vec<T, N> vwggid(wggid), vsgid(sgid);
        if (wggid == 0)
          sgsizeacc[0] = SG.get_max_local_range()[0];

        /*GID of middle element in every subgroup*/
        acc_select[NdItem.get_global_id()] =
            select_from_group(SG, vwggid, SG.get_max_local_range()[0] / 2);
        /* Save GID with SGLID = ( SGLID XOR SGID ) % SGMaxSize */
        acc_xor[NdItem.get_global_id()] = permute_group_by_xor(
            SG, vwggid, sgid % SG.get_max_local_range()[0]);
      });
    });
    host_accessor acc_select{buf_select, sycl::read_write};
    host_accessor acc_xor{buf_xor, sycl::read_write};
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
      /*GID of middle element in every subgroup*/
      exit_if_not_equal_vec<T, N>(
          acc_select[j], vec<T, N>(j / L * L + SGid * sg_size + sg_size / 2),
          "select_from_group");
      /* Value GID with SGLID = ( SGLID XOR SGID ) % SGMaxSize */
      exit_if_not_equal_vec(acc_xor[j],
                            vec<T, N>(SGBeginGid + (SGLid ^ (SGid % sg_size))),
                            "permute_group_by_xor");
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
    buffer<T> buf_select(G);
    buffer<T> buf_xor(G);
    buffer<size_t> sgsizebuf(1);
    Queue.submit([&](handler &cgh) {
      accessor acc_select{buf_select, cgh, sycl::read_write};
      accessor acc_xor{buf_xor, cgh, sycl::read_write};
      accessor sgsizeacc{sgsizebuf, cgh, sycl::read_write};
      cgh.parallel_for<sycl_subgr<T, 0>>(NdRange, [=](nd_item<1> NdItem) {
        ext::oneapi::sub_group SG = NdItem.get_sub_group();
        uint32_t wggid = NdItem.get_global_id(0);
        uint32_t sgid = SG.get_group_id().get(0);
        if (wggid == 0)
          sgsizeacc[0] = SG.get_max_local_range()[0];

        /*GID of middle element in every subgroup*/
        acc_select[NdItem.get_global_id()] =
            select_from_group(SG, wggid, SG.get_max_local_range()[0] / 2);
        /* Save GID with SGLID = ( SGLID XOR SGID ) % SGMaxSize */
        acc_xor[NdItem.get_global_id()] =
            permute_group_by_xor(SG, wggid, sgid % SG.get_max_local_range()[0]);
      });
    });
    host_accessor acc_select{buf_select, sycl::read_write};
    host_accessor acc_xor{buf_xor, sycl::read_write};
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

      /*GID of middle element in every subgroup*/
      exit_if_not_equal<T>(acc_select[j],
                           j / L * L + SGid * sg_size + sg_size / 2,
                           "select_from_group");

      /* Value GID with SGLID = ( SGLID XOR SGID ) % SGMaxSize */
      exit_if_not_equal<T>(acc_xor[j], SGBeginGid + (SGLid ^ (SGid % sg_size)),
                           "permute_group_by_xor");
      SGLid++;
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }
}
