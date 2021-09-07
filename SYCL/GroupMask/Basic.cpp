// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// REQUIRES: gpu
// UNSUPPORTED: cuda, hip
// GroupNonUniformBallot capability is supported on Intel GPU only
// RUN: %GPU_RUN_PLACEHOLDER %t.out

//==---------- Basic.cpp - SYCL Group Mask basic test ----------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
using namespace sycl;
constexpr int global_size = 128;
constexpr int local_size = 32;
int main() {
#ifdef SYCL_EXT_ONEAPI_GROUP_MASK
  queue Queue;

  try {
    nd_range<1> NdRange(global_size, local_size);
    int Res = 0;
    {
      buffer resbuf(&Res, range<1>(1));

      Queue.submit([&](handler &cgh) {
        auto resacc = resbuf.get_access<access::mode::read_write>(cgh);

        cgh.parallel_for<class group_mask>(NdRange, [=](nd_item<1> NdItem) {
          size_t GID = NdItem.get_global_linear_id();
          auto SG = NdItem.get_sub_group();
          auto gmask_gid2 =
              ext::oneapi::group_ballot(NdItem.get_sub_group(), GID % 2);
          auto gmask_gid3 =
              ext::oneapi::group_ballot(NdItem.get_sub_group(), GID % 3);
          NdItem.barrier();

          if (!GID) {
            int res = 0;

            for (size_t i = 0; i < SG.get_max_local_range()[0]; i++) {
              res |= !((gmask_gid2 | gmask_gid3)[i] == (i % 2 || i % 3)) << 1;
              res |= !((gmask_gid2 & gmask_gid3)[i] == (i % 2 && i % 3)) << 2;
              res |= !((gmask_gid2 ^ gmask_gid3)[i] ==
                       ((bool)(i % 2) ^ (bool)(i % 3)))
                     << 3;
            }
            gmask_gid2 <<= 32;
            res |= (gmask_gid2.extract_bits()[2] != 0xaaaaaaaa) << 4;
            res |= ((gmask_gid2 >> 8).extract_bits()[3] != 0xaa000000) << 5;
            res |= ((gmask_gid3 >> 8).extract_bits()[3] != 0xb6db6d) << 6;
            res |= (!gmask_gid2[32] && gmask_gid2[31]) << 7;
            gmask_gid3[0] = gmask_gid3[3] = gmask_gid3[6] = true;
            res |= (gmask_gid3.extract_bits()[3] != 0xb6db6dff) << 7;
            gmask_gid3.reset();
            res |= !(gmask_gid3.none() && gmask_gid2.any() && !gmask_gid2.all())
                   << 8;
            gmask_gid2.set();
            res |= !(gmask_gid3.none() && gmask_gid2.any() && gmask_gid2.all())
                   << 9;
            gmask_gid3.flip();
            res |= (gmask_gid3 != gmask_gid2) << 10;
            resacc[0] = res;
          }
        });
      });
    }
    if (Res) {
      std::cout << "Unexpected result for group_mask operation: " << Res
                << std::endl;
      exit(1);
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }

  std::cout << "Test passed." << std::endl;
#else
  std::cout << "Test skipped due to missing extension." << std::endl;
#endif
  return 0;
}
