// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
//==-- generic_shuffle.cpp - SYCL sub_group generic shuffle test *- C++ -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helper.hpp"
#include <algorithm>
#include <complex>
#include <sycl/sycl.hpp>
#include <vector>
template <typename T> class pointer_kernel;

using namespace sycl;

template <typename SpecializationKernelName, typename T>
void check_pointer(queue &Queue, size_t G = 256, size_t L = 64) {
  try {
    nd_range<1> NdRange(G, L);
    buffer<T *> buf(G);
    buffer<T *> buf_up(G);
    buffer<T *> buf_down(G);
    buffer<T *> buf_xor(G);
    buffer<size_t> sgsizebuf(1);
    Queue.submit([&](handler &cgh) {
      auto acc = buf.template get_access<access::mode::read_write>(cgh);
      auto acc_up = buf_up.template get_access<access::mode::read_write>(cgh);
      auto acc_down =
          buf_down.template get_access<access::mode::read_write>(cgh);
      auto acc_xor = buf_xor.template get_access<access::mode::read_write>(cgh);
      auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>(cgh);

      cgh.parallel_for<SpecializationKernelName>(
          NdRange, [=](nd_item<1> NdItem) {
            ext::oneapi::sub_group SG = NdItem.get_sub_group();
            uint32_t wggid = NdItem.get_global_id(0);
            uint32_t sgid = SG.get_group_id().get(0);
            if (wggid == 0)
              sgsizeacc[0] = SG.get_max_local_range()[0];

            T *ptr = static_cast<T *>(0x0) + wggid;

            /*GID of middle element in every subgroup*/
            acc[NdItem.get_global_id()] =
                SG.shuffle(ptr, SG.get_max_local_range()[0] / 2);

            /* Save GID-SGID */
            acc_up[NdItem.get_global_id()] = SG.shuffle_up(ptr, sgid);

            /* Save GID+SGID */
            acc_down[NdItem.get_global_id()] = SG.shuffle_down(ptr, sgid);

            /* Save GID with SGLID = ( SGLID XOR SGID ) % SGMaxSize */
            acc_xor[NdItem.get_global_id()] =
                SG.shuffle_xor(ptr, sgid % SG.get_max_local_range()[0]);
          });
    });
    auto acc = buf.template get_access<access::mode::read_write>();
    auto acc_up = buf_up.template get_access<access::mode::read_write>();
    auto acc_down = buf_down.template get_access<access::mode::read_write>();
    auto acc_xor = buf_xor.template get_access<access::mode::read_write>();
    auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>();

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
      exit_if_not_equal(acc[j],
                        static_cast<T *>(0x0) +
                            (j / L * L + SGid * sg_size + sg_size / 2),
                        "shuffle");

      /* Value GID+SGID for all element except last SGID in SG*/
      if (j % L % sg_size + SGid < sg_size && j % L + SGid < L) {
        exit_if_not_equal(acc_down[j], static_cast<T *>(0x0) + (j + SGid),
                          "shuffle_down");
      }

      /* Value GID-SGID for all element except first SGID in SG*/
      if (j % L % sg_size >= SGid) {
        exit_if_not_equal(acc_up[j], static_cast<T *>(0x0) + (j - SGid),
                          "shuffle_up");
      }

      /* Value GID with SGLID = ( SGLID XOR SGID ) % SGMaxSize */
      exit_if_not_equal(acc_xor[j],
                        static_cast<T *>(0x0) +
                            (SGBeginGid + (SGLid ^ (SGid % sg_size))),
                        "shuffle_xor");
      SGLid++;
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }
}

template <typename SpecializationKernelName, typename T, typename Generator>
void check_struct(queue &Queue, Generator &Gen, size_t G = 256, size_t L = 64) {

  // Fill a vector with values that will be shuffled
  std::vector<T> values(G);
  std::generate(values.begin(), values.end(), Gen);

  try {
    nd_range<1> NdRange(G, L);
    buffer<T> buf(G);
    buffer<T> buf_up(G);
    buffer<T> buf_down(G);
    buffer<T> buf_xor(G);
    buffer<size_t> sgsizebuf(1);
    buffer<T> buf_in(values.data(), values.size());
    Queue.submit([&](handler &cgh) {
      auto acc = buf.template get_access<access::mode::read_write>(cgh);
      auto acc_up = buf_up.template get_access<access::mode::read_write>(cgh);
      auto acc_down =
          buf_down.template get_access<access::mode::read_write>(cgh);
      auto acc_xor = buf_xor.template get_access<access::mode::read_write>(cgh);
      auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>(cgh);
      auto in = buf_in.template get_access<access::mode::read>(cgh);

      cgh.parallel_for<SpecializationKernelName>(
          NdRange, [=](nd_item<1> NdItem) {
            ext::oneapi::sub_group SG = NdItem.get_sub_group();
            uint32_t wggid = NdItem.get_global_id(0);
            uint32_t sgid = SG.get_group_id().get(0);
            if (wggid == 0)
              sgsizeacc[0] = SG.get_max_local_range()[0];

            T val = in[wggid];

            /*GID of middle element in every subgroup*/
            acc[NdItem.get_global_id()] =
                SG.shuffle(val, SG.get_max_local_range()[0] / 2);

            /* Save GID-SGID */
            acc_up[NdItem.get_global_id()] = SG.shuffle_up(val, sgid);

            /* Save GID+SGID */
            acc_down[NdItem.get_global_id()] = SG.shuffle_down(val, sgid);

            /* Save GID with SGLID = ( SGLID XOR SGID ) % SGMaxSize */
            acc_xor[NdItem.get_global_id()] =
                SG.shuffle_xor(val, sgid % SG.get_max_local_range()[0]);
          });
    });
    auto acc = buf.template get_access<access::mode::read_write>();
    auto acc_up = buf_up.template get_access<access::mode::read_write>();
    auto acc_down = buf_down.template get_access<access::mode::read_write>();
    auto acc_xor = buf_xor.template get_access<access::mode::read_write>();
    auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>();

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
      exit_if_not_equal(
          acc[j], values[j / L * L + SGid * sg_size + sg_size / 2], "shuffle");

      /* Value GID+SGID for all element except last SGID in SG*/
      if (j % L % sg_size + SGid < sg_size && j % L + SGid < L) {
        exit_if_not_equal(acc_down[j], values[j + SGid], "shuffle_down");
      }

      /* Value GID-SGID for all element except first SGID in SG*/
      if (j % L % sg_size >= SGid) {
        exit_if_not_equal(acc_up[j], values[j - SGid], "shuffle_up");
      }

      /* Value GID with SGLID = ( SGLID XOR SGID ) % SGMaxSize */
      exit_if_not_equal(acc_xor[j],
                        values[SGBeginGid + (SGLid ^ (SGid % sg_size))],
                        "shuffle_xor");
      SGLid++;
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }
}

int main() {
  queue Queue;
  if (Queue.get_device().is_host()) {
    std::cout << "Skipping test\n";
    return 0;
  }

  // Test shuffle of pointer types
  check_pointer<class KernelName_mNiN, int>(Queue);

  // Test shuffle of non-native types
  auto ComplexFloatGenerator = [state = std::complex<float>(0, 1)]() mutable {
    return state += std::complex<float>(2, 2);
  };
  check_struct<class KernelName_zHfIPOLOFsXiZiCvG, std::complex<float>>(
      Queue, ComplexFloatGenerator);

  auto ComplexDoubleGenerator = [state = std::complex<double>(0, 1)]() mutable {
    return state += std::complex<double>(2, 2);
  };
  check_struct<class KernelName_CjlHUmnuxWtyejZFD, std::complex<double>>(
      Queue, ComplexDoubleGenerator);

  std::cout << "Test passed." << std::endl;
  return 0;
}
