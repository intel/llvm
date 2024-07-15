//==------------ shuffle.hpp - SYCL sub_group shuffle test -----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helper.hpp"
#include <iostream>
template <typename T, int N> class sycl_subgr;

using namespace sycl;

// TODO remove this workaround when clang will support correct generation of
// half typename in integration header
struct wa_half;

template <typename T, int N>
void check(queue &Queue, size_t G = 256, size_t L = 64) {
  try {
    nd_range<1> NdRange(G, L);
    buffer<vec<T, N>> buf(G);
    buffer<vec<T, N>> buf_up(G);
    buffer<vec<T, N>> buf_down(G);
    buffer<vec<T, N>> buf_xor(G);
    buffer<size_t> sgsizebuf(1);
    Queue.submit([&](handler &cgh) {
      auto acc = buf.template get_access<access::mode::read_write>(cgh);
      auto acc_up = buf_up.template get_access<access::mode::read_write>(cgh);
      auto acc_down =
          buf_down.template get_access<access::mode::read_write>(cgh);
      auto acc_xor = buf_xor.template get_access<access::mode::read_write>(cgh);
      auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>(cgh);

      cgh.parallel_for<sycl_subgr<T, N>>(NdRange, [=](nd_item<1> NdItem) {
        sycl::sub_group SG = NdItem.get_sub_group();
        uint32_t wggid = NdItem.get_global_id(0);
        uint32_t sgid = SG.get_group_id().get(0);
        vec<T, N> vwggid(wggid), vsgid(sgid);
        if (wggid == 0)
          sgsizeacc[0] = SG.get_max_local_range()[0];

        /*GID of middle element in every subgroup*/
        acc[NdItem.get_global_id()] =
            SG.shuffle(vwggid, SG.get_max_local_range()[0] / 2);
        /* Save GID-SGID */
        acc_up[NdItem.get_global_id()] = SG.shuffle_up(vwggid, sgid);
        /* Save GID+SGID */
        acc_down[NdItem.get_global_id()] = SG.shuffle_down(vwggid, sgid);
        /* Save GID with SGLID = ( SGLID XOR SGID ) % SGMaxSize */
        acc_xor[NdItem.get_global_id()] =
            SG.shuffle_xor(vwggid, sgid % SG.get_max_local_range()[0]);
      });
    });
    host_accessor acc(buf);
    host_accessor acc_up(buf_up);
    host_accessor acc_down(buf_down);
    host_accessor acc_xor(buf_xor);
    host_accessor sgsizeacc(sgsizebuf);

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
          acc[j], vec<T, N>(j / L * L + SGid * sg_size + sg_size / 2),
          "shuffle");
      /* Value GID+SGID for all element except last SGID in SG*/
      if (j % L % sg_size + SGid < sg_size && j % L + SGid < L) {
        exit_if_not_equal_vec(acc_down[j], vec<T, N>(j + SGid % sg_size),
                              "shuffle_down");
      }
      /* Value GID-SGID for all element except first SGID in SG*/
      if (j % L % sg_size >= SGid) {
        exit_if_not_equal_vec(acc_up[j], vec<T, N>(j - SGid % sg_size),
                              "shuffle_up");
      }
      /* Value GID with SGLID = ( SGLID XOR SGID ) % SGMaxSize */
      exit_if_not_equal_vec(acc_xor[j],
                            vec<T, N>(SGBeginGid + (SGLid ^ (SGid % sg_size))),
                            "shuffle_xor");
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
    buffer<T> buf(G);
    buffer<T> buf_up(G);
    buffer<T> buf_down(G);
    buffer<T> buf_xor(G);
    buffer<size_t> sgsizebuf(1);
    Queue.submit([&](handler &cgh) {
      auto acc = buf.template get_access<access::mode::read_write>(cgh);
      auto acc_up = buf_up.template get_access<access::mode::read_write>(cgh);
      auto acc_down =
          buf_down.template get_access<access::mode::read_write>(cgh);
      auto acc_xor = buf_xor.template get_access<access::mode::read_write>(cgh);
      auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>(cgh);

      cgh.parallel_for<sycl_subgr<T, 0>>(NdRange, [=](nd_item<1> NdItem) {
        sycl::sub_group SG = NdItem.get_sub_group();
        uint32_t wggid = NdItem.get_global_id(0);
        uint32_t sgid = SG.get_group_id().get(0);
        if (wggid == 0)
          sgsizeacc[0] = SG.get_max_local_range()[0];

        /*GID of middle element in every subgroup*/
        acc[NdItem.get_global_id()] =
            SG.shuffle<T>(wggid, SG.get_max_local_range()[0] / 2);
        /* Save GID-SGID */
        acc_up[NdItem.get_global_id()] = SG.shuffle_up<T>(wggid, sgid);
        /* Save GID+SGID */
        acc_down[NdItem.get_global_id()] = SG.shuffle_down<T>(wggid, sgid);
        /* Save GID with SGLID = ( SGLID XOR SGID ) % SGMaxSize */
        acc_xor[NdItem.get_global_id()] =
            SG.shuffle_xor<T>(wggid, sgid % SG.get_max_local_range()[0]);
      });
    });
    host_accessor acc(buf);
    host_accessor acc_up(buf_up);
    host_accessor acc_down(buf_down);
    host_accessor acc_xor(buf_xor);
    host_accessor sgsizeacc(sgsizebuf);

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
      exit_if_not_equal<T>(acc[j], j / L * L + SGid * sg_size + sg_size / 2,
                           "shuffle");
      /* Value GID+SGID for all element except last SGID in SG*/
      if (j % L % sg_size + SGid < sg_size && j % L + SGid < L) {
        exit_if_not_equal<T>(acc_down[j], j + SGid, "shuffle_down");
      }
      /* Value GID-SGID for all element except first SGID in SG*/
      if (j % L % sg_size >= SGid) {
        exit_if_not_equal<T>(acc_up[j], j - SGid, "shuffle_up");
      }
      /* Value GID with SGLID = ( SGLID XOR SGID ) % SGMaxSize */
      exit_if_not_equal<T>(acc_xor[j], SGBeginGid + (SGLid ^ (SGid % sg_size)),
                           "shuffle_xor");
      SGLid++;
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }
}
