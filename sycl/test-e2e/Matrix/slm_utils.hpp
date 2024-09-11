//==------------------ slm_utils.hpp  - DPC++ joint_matrix------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
template <unsigned int colsA, unsigned int colsB, unsigned int MCache2,
          unsigned int NCache2, unsigned int KCache2, unsigned int vnniFactor,
          size_t SGs, typename TOperand, access::address_space space>
inline void
slm_read_write(multi_ptr<TOperand, space, access::decorated::yes> pA,
               multi_ptr<TOperand, space, access::decorated::yes> pB,
               local_accessor<TOperand, 2> tileA,
               local_accessor<TOperand, 2> tileB, sub_group sg, unsigned int k2,
               size_t m2, size_t n2, size_t sgSize) {
  // Number of elements to be loaded into SLM per WI
  size_t elemsPerLoadA = KCache2 / sgSize;
  for (int i = 0; i < MCache2 / SGs; i++) {
    size_t GlOffsetA =
        (m2 * MCache2 + sg.get_group_id() * (MCache2 / SGs) + i) * colsA +
        k2 * KCache2;
    size_t LocOffsetA = (sg.get_group_id() * (MCache2 / SGs) + i) * KCache2;

    if (elemsPerLoadA == 2) {
      vec<TOperand, 2> slmVecA = sg.load<2>(pA + GlOffsetA);
      sg.store<2>(tileA.template get_multi_ptr<sycl::access::decorated::no>() +
                      LocOffsetA,
                  slmVecA);
    } else if (elemsPerLoadA == 4) {
      vec<TOperand, 4> slmVecA = sg.load<4>(pA + GlOffsetA);
      sg.store<4>(tileA.template get_multi_ptr<sycl::access::decorated::no>() +
                      LocOffsetA,
                  slmVecA);
    } else if (elemsPerLoadA == 1) {
      TOperand slmScaA = sg.load(pA + GlOffsetA);

      sg.store(tileA.template get_multi_ptr<sycl::access::decorated::no>() +
                   LocOffsetA,
               slmScaA);
    } else
      assert(elemsPerLoadA == 1 || elemsPerLoadA == 2 || elemsPerLoadA == 4);
  }
  // how much each SG will load to SLM --> has to be contiguous
  // NCache2*KCache2/(SGs*SG_SIZE) = 16
  size_t elemsPerLoadB = NCache2 * KCache2 / (SGs * sgSize);
  size_t sgsPerRow = (NCache2 * vnniFactor) / (elemsPerLoadB * sgSize);
  size_t GlOffsetB =
      (k2 * (KCache2 / vnniFactor) + (uint)(sg.get_group_id() / sgsPerRow)) *
          (colsB * vnniFactor) +
      n2 * NCache2 * vnniFactor +
      (sg.get_group_id() % sgsPerRow) * (elemsPerLoadB * sgSize);
  size_t LocOffsetB =
      ((uint)(sg.get_group_id() / sgsPerRow)) * NCache2 * vnniFactor +
      (sg.get_group_id() % sgsPerRow) * elemsPerLoadB * sgSize;
  if (elemsPerLoadB == 16) {
    vec<TOperand, 16> slmVecB = sg.load<16>(pB + GlOffsetB);

    sg.store<16>(tileB.template get_multi_ptr<sycl::access::decorated::no>() +
                     LocOffsetB,
                 slmVecB);
  } else if (elemsPerLoadB == 8) {
    vec<TOperand, 8> slmVecB = sg.load<8>(pB + GlOffsetB);

    sg.store<8>(tileB.template get_multi_ptr<sycl::access::decorated::no>() +
                    LocOffsetB,
                slmVecB);
  } else
    assert(elemsPerLoadB == 8 || elemsPerLoadB == 16);
}
