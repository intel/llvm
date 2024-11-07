//==------------------ slm_utils.hpp  - DPC++ joint_matrix------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/ext/oneapi/experimental/group_load_store.hpp>

template <unsigned int colsA, unsigned int colsB, unsigned int MCache2,
          unsigned int NCache2, unsigned int KCache2, unsigned int vnniFactor,
          size_t SGs, typename TOperand, access::address_space space>
inline void
slm_read_write(multi_ptr<TOperand, space, access::decorated::yes> pA,
               multi_ptr<TOperand, space, access::decorated::yes> pB,
               local_accessor<TOperand, 2> tileA,
               local_accessor<TOperand, 2> tileB, sub_group sg, unsigned int k2,
               size_t m2, size_t n2, size_t sgSize) {
  using namespace sycl::ext::oneapi::experimental;
  // Number of elements to be loaded into SLM per WI
  size_t elemsPerLoadA = KCache2 / sgSize;
  for (int i = 0; i < MCache2 / SGs; i++) {
    size_t GlOffsetA =
        (m2 * MCache2 + sg.get_group_id() * (MCache2 / SGs) + i) * colsA +
        k2 * KCache2;
    size_t LocOffsetA = (sg.get_group_id() * (MCache2 / SGs) + i) * KCache2;

    auto SrcA = pA + GlOffsetA;
    auto DstA = tileA.template get_multi_ptr<sycl::access::decorated::no>() +
                LocOffsetA;

    if (elemsPerLoadA == 2) {
      vec<TOperand, 2> slmVecA;
      group_load(sg, SrcA, slmVecA, properties(data_placement_striped));
      group_store(sg, slmVecA, DstA, properties(data_placement_striped));
    } else if (elemsPerLoadA == 4) {
      vec<TOperand, 4> slmVecA;
      group_load(sg, SrcA, slmVecA, properties(data_placement_striped));
      group_store(sg, slmVecA, DstA, properties(data_placement_striped));
    } else if (elemsPerLoadA == 1) {
      vec<TOperand, 1> slmVecA;
      group_load(sg, SrcA, slmVecA, properties(data_placement_striped));
      group_store(sg, slmVecA, DstA, properties(data_placement_striped));
    } else
      assert(elemsPerLoadA == 1 || elemsPerLoadA == 2 || elemsPerLoadA == 4);
  }
  // how much each SG will load to SLM --> has to be contiguous
  // NCache2*KCache2/(SGs*SG_SIZE) = 16
  size_t elemsPerLoadB = NCache2 * KCache2 / (SGs * sgSize);
  size_t sgsPerRow = (NCache2 * vnniFactor) / (elemsPerLoadB * sgSize);
  size_t GlOffsetB = (k2 * (KCache2 / vnniFactor) +
                      (unsigned int)(sg.get_group_id() / sgsPerRow)) *
                         (colsB * vnniFactor) +
                     n2 * NCache2 * vnniFactor +
                     (sg.get_group_id() % sgsPerRow) * (elemsPerLoadB * sgSize);
  size_t LocOffsetB =
      ((unsigned int)(sg.get_group_id() / sgsPerRow)) * NCache2 * vnniFactor +
      (sg.get_group_id() % sgsPerRow) * elemsPerLoadB * sgSize;
  auto SrcB = pB + GlOffsetB;
  auto DstB =
      tileB.template get_multi_ptr<sycl::access::decorated::no>() + LocOffsetB;
  if (elemsPerLoadB == 16) {
    vec<TOperand, 16> slmVecB;
    group_load(sg, SrcB, slmVecB, properties(data_placement_striped));
    group_store(sg, slmVecB, DstB, properties(data_placement_striped));
  } else if (elemsPerLoadB == 8) {
    vec<TOperand, 8> slmVecB;
    group_load(sg, SrcB, slmVecB, properties(data_placement_striped));
    group_store(sg, slmVecB, DstB, properties(data_placement_striped));
  } else
    assert(elemsPerLoadB == 8 || elemsPerLoadB == 16);
}
