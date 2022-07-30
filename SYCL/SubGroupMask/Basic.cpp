// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// REQUIRES: gpu
// UNSUPPORTED: hip
// GroupNonUniformBallot capability is supported on Intel GPU only
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// UNSUPPORTED: ze_debug-1,ze_debug4

//==---------- Basic.cpp - sub-group mask basic test -----------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;
constexpr int global_size = 128;
constexpr int local_size = 32;
int main() {
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK
  queue Queue;

  try {
    nd_range<1> NdRange(global_size, local_size);
    int Res = 0;
    {
      buffer resbuf(&Res, range<1>(1));

      Queue.submit([&](handler &cgh) {
        auto resacc = resbuf.get_access<access::mode::read_write>(cgh);

        cgh.parallel_for<class sub_group_mask_test>(
            NdRange, [=](nd_item<1> NdItem) [[intel::reqd_sub_group_size(32)]] {
              size_t GID = NdItem.get_global_linear_id();
              auto SG = NdItem.get_sub_group();
              // AAAAAAAA
              auto gmask_gid2 =
                  ext::oneapi::group_ballot(NdItem.get_sub_group(), GID % 2);
              // B6DB6DB6
              auto gmask_gid3 =
                  ext::oneapi::group_ballot(NdItem.get_sub_group(), GID % 3);

              if (!GID) {
                int res = 0;

                for (size_t i = 0; i < SG.get_max_local_range()[0]; i++) {
                  res |= !((gmask_gid2 | gmask_gid3)[i] == (i % 2 || i % 3))
                         << 1;
                  res |= !((gmask_gid2 & gmask_gid3)[i] == (i % 2 && i % 3))
                         << 2;
                  res |= !((gmask_gid2 ^ gmask_gid3)[i] ==
                           ((bool)(i % 2) ^ (bool)(i % 3)))
                         << 3;
                }
                gmask_gid2 <<= 8;
                uint32_t r = 0;
                gmask_gid2.extract_bits(r);
                res |= (r != 0xaaaaaa00) << 4;
                (gmask_gid2 >> 4).extract_bits(r);
                res |= (r != 0x0aaaaaa0) << 5;
                gmask_gid3.insert_bits((char)0b01010101, 8);
                res |= (!gmask_gid3[8] || gmask_gid3[9] || !gmask_gid3[10] ||
                        gmask_gid3[11])
                       << 6;
                marray<unsigned char, 6> mr{1};
                gmask_gid3.extract_bits(mr);
                res |= (mr[0] != 0xb6 || mr[1] != 0x55 || mr[2] != 0xdb ||
                        mr[3] != 0xb6 || mr[4] || mr[5])
                       << 7;
                res |= (gmask_gid2[30] || !gmask_gid2[31]) << 8;
                gmask_gid3[0] = gmask_gid3[3] = gmask_gid3[6] = true;
                gmask_gid3.extract_bits(r);
                res |= (r != 0xb6db55ff) << 9;
                gmask_gid3.reset();
                res |= !(gmask_gid3.none() && gmask_gid2.any() &&
                         !gmask_gid2.all())
                       << 10;
                gmask_gid2.set();
                res |=
                    !(gmask_gid3.none() && gmask_gid2.any() && gmask_gid2.all())
                    << 11;
                gmask_gid3.flip();
                res |= (gmask_gid3 != gmask_gid2) << 12;
                resacc[0] = res;
              }
            });
      });
    }
    if (Res) {
      std::cout << "Unexpected result for sub_group_mask operation: " << Res
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
