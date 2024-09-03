//==------------------ slm_utils.hpp  - DPC++ joint_matrix------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
template <unsigned int colsA, unsigned int colsB, unsigned int MCache2,
          unsigned int NCache2, unsigned int KCache2, unsigned int vnniFactor,
          size_t SGs, typename TOperand, access::address_space Space>
inline void
slm_read_write(multi_ptr<TOperand, Space, access::decorated::yes> pA,
               multi_ptr<TOperand, Space, access::decorated::yes> pB,
               local_accessor<TOperand, 2> tileA,
               local_accessor<TOperand, 2> tileB, sub_group sg, unsigned int k2,
               size_t m2, size_t n2, size_t sgSize) {
  size_t slmReadA = KCache2 / sgSize;
  for (int i = 0; i < MCache2 / SGs; i++) {
    if (slmReadA == 2) {
      vec<TOperand, 2> slmVec = sg.load<2>(
          pA +
          (m2 * MCache2 + (sg.get_group_id()) * (MCache2 / SGs) + i) * colsA +
          k2 * KCache2);

      sg.store<2>(tileA.template get_multi_ptr<sycl::access::decorated::no>() +
                      ((sg.get_group_id()) * (MCache2 / SGs) + i) * KCache2,
                  slmVec);
    } else if (slmReadA == 4) {
      vec<TOperand, 4> slmVec = sg.load<4>(
          pA +
          (m2 * MCache2 + (sg.get_group_id()) * (MCache2 / SGs) + i) * colsA +
          k2 * KCache2);

      sg.store<4>(tileA.template get_multi_ptr<sycl::access::decorated::no>() +
                      ((sg.get_group_id()) * (MCache2 / SGs) + i) * KCache2,
                  slmVec);
    } else if (slmReadA == 1) {
      TOperand slmS = sg.load(
          pA +
          (m2 * MCache2 + (sg.get_group_id()) * (MCache2 / SGs) + i) * colsA +
          k2 * KCache2);

      sg.store(tileA.template get_multi_ptr<sycl::access::decorated::no>() +
                   ((sg.get_group_id()) * (MCache2 / SGs) + i) * KCache2,
               slmS);
    } else
      assert(slmReadA == 1 || slmReadA == 2 || slmReadA == 4);
  }
  // how much each SG will load to SLM --> has to be contiguous
  // NCache2*KCache2/(SGs*SG_SIZE) = 16
  size_t slmRead = NCache2 * KCache2 / (SGs * sgSize);
  size_t sgsPerRow = (NCache2 * vnniFactor) / (slmRead * sgSize);
  if (slmRead == 16) {
    vec<TOperand, 16> slmvecb = sg.load<16>(
        pB +
        (k2 * (KCache2 / vnniFactor) + (uint)(sg.get_group_id() / sgsPerRow)) *
            (colsB * vnniFactor) +
        n2 * NCache2 * vnniFactor +
        (sg.get_group_id() % sgsPerRow) * (slmRead * sgSize));

    sg.store<16>(tileB.template get_multi_ptr<sycl::access::decorated::no>() +
                     ((uint)(sg.get_group_id() / sgsPerRow)) * NCache2 *
                         vnniFactor +
                     (sg.get_group_id() % sgsPerRow) * (slmRead * sgSize),
                 slmvecb);
  } else if (slmRead == 8) {
    vec<TOperand, 8> slmvecb = sg.load<8>(
        pB +
        (k2 * (KCache2 / vnniFactor) + (uint)(sg.get_group_id() / sgsPerRow)) *
            (colsB * vnniFactor) +
        n2 * NCache2 * vnniFactor +
        (sg.get_group_id() % sgsPerRow) * (slmRead * sgSize));

    sg.store<8>(tileB.template get_multi_ptr<sycl::access::decorated::no>() +
                    ((uint)(sg.get_group_id() / sgsPerRow)) * NCache2 *
                        vnniFactor +
                    (sg.get_group_id() % sgsPerRow) * (slmRead * sgSize),
                slmvecb);
  } else
    assert(slmRead == 8 || slmRead == 16);
}
