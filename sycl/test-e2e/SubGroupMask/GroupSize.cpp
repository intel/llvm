// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-device-code-split=per_kernel %s -o %t.out
// REQUIRES: gpu
// UNSUPPORTED: hip
// GroupNonUniformBallot capability is supported on Intel GPU only
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// UNSUPPORTED: ze_debug-1,ze_debug4

//==- GroupSize.cpp - sub-group mask dependency on group size --*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK
constexpr int global_size = 128;
constexpr int local_size = 32;
template <size_t> class sycl_subgr;

template <size_t SGSize> void test(queue Queue) {
  std::cout << "Testing sub_group_mask for sub-group size=" << SGSize
            << std::endl;
  try {
    nd_range<1> NdRange(global_size, local_size);
    int Res[32 / SGSize] = {0};
    {
      buffer resbuf(Res, range<1>(32 / SGSize));

      Queue.submit([&](handler &cgh) {
        auto resacc = resbuf.template get_access<access::mode::read_write>(cgh);

        cgh.parallel_for<sycl_subgr<SGSize>>(
            NdRange, [=
        ](nd_item<1> NdItem) [[intel::reqd_sub_group_size(SGSize)]] {
              auto SG = NdItem.get_sub_group();
              auto LID = SG.get_local_id();
              auto SGID = SG.get_group_id();

              auto gmask_gid2 =
                  ext::oneapi::group_ballot(NdItem.get_sub_group(), LID % 2);
              auto gmask_gid3 =
                  ext::oneapi::group_ballot(NdItem.get_sub_group(), LID % 3);

              if (!LID) {
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
                res |= (gmask_gid2.size() != SG.get_max_local_range()[0]) << 4;
                resacc[SGID] = res;
              }
            });
      });
    }
    for (size_t i = 0; i < 32 / SGSize; i++) {
      if (Res[i]) {
        std::cout
            << "Unexpected result for sub_group_mask operation for sub-group "
            << i << ": " << Res[i] << std::endl;
        exit(1);
      }
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }
}
#endif // SYCL_EXT_ONEAPI_SUB_GROUP_MASK

int main() {
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK
  queue Queue;

  for (const auto &x :
       Queue.get_device().get_info<info::device::sub_group_sizes>()) {

    switch (x) {
    case 8:
      test<8>(Queue);
      break;
    case 16:
      test<16>(Queue);
      break;
    case 32:
      test<32>(Queue);
      break;
    default:
      std::cout << "Sub group size of " << x << " supported, but not tested."
                << std::endl;
      break;
    }
  }

  std::cout << "Test passed." << std::endl;
#else
  std::cout << "Test skipped due to missing extension." << std::endl;
#endif
  return 0;
}
