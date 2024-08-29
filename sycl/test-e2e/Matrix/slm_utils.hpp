//==------------------ slm_utils.hpp  - DPC++ joint_matrix------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifdef OCL
typedef ushort ushort16ocl __attribute__((ext_vector_type(16)));
typedef ushort ushort8ocl __attribute__((ext_vector_type(8)));
typedef ushort ushort4ocl __attribute__((ext_vector_type(4)));
typedef ushort ushort2ocl __attribute__((ext_vector_type(2)));

extern "C" {
SYCL_EXTERNAL ushort2ocl __builtin_IB_simd_block_read_2_global_h(
    __attribute__((opencl_global)) ushort *base);
SYCL_EXTERNAL void __builtin_IB_simd_block_write_2_local_h(
    __attribute__((opencl_local)) ushort *base, ushort2ocl);
SYCL_EXTERNAL ushort4ocl __builtin_IB_simd_block_read_4_global_h(
    __attribute__((opencl_global)) ushort *base);
SYCL_EXTERNAL void __builtin_IB_simd_block_write_4_local_h(
    __attribute__((opencl_local)) ushort *base, ushort4ocl);
SYCL_EXTERNAL ushort8ocl __builtin_IB_simd_block_read_8_global_h(
    __attribute__((opencl_global)) ushort *base);
SYCL_EXTERNAL void __builtin_IB_simd_block_write_8_local_h(
    __attribute__((opencl_local)) ushort *base, ushort8ocl);
SYCL_EXTERNAL ushort16ocl __builtin_IB_simd_block_read_16_global_h(
    __attribute__((opencl_global)) ushort *base);
SYCL_EXTERNAL void __builtin_IB_simd_block_write_16_local_h(
    __attribute__((opencl_local)) ushort *base, ushort16ocl);
};

template <unsigned int rowsA, unsigned int colsA, unsigned int rowsB,
          unsigned int colsB, unsigned int MCACHE2, unsigned int NCACHE2,
          unsigned int KCACHE2, unsigned int vnniFactor, size_t SGs,
          typename TOperand>
inline void
slm_read_write_OCL(TOperand *A, TOperand *B, local_accessor<TOperand, 2> tileA,
                   local_accessor<TOperand, 2> tileB, sub_group sg,
                   unsigned int k2, size_t m2, size_t n2, size_t SG_SIZE) {

  size_t slmReadA = KCACHE2 / SG_SIZE;

  for (int i = 0; i < (MCACHE2 / SGs); i++) {
    __attribute__((opencl_global)) ushort *g_addrA;
    g_addrA =
        (__attribute__((opencl_global))
         ushort *)(A +
                   (m2 * MCACHE2 + (sg.get_group_id()) * (MCACHE2 / SGs) + i) *
                       colsA +
                   k2 * KCACHE2);
    __attribute__((opencl_local)) ushort *l_addrA;
    l_addrA =
        (__attribute__((opencl_local)) ushort
             *)(tileA.template get_multi_ptr<sycl::access::decorated::yes>() +
                ((sg.get_group_id()) * (MCACHE2 / SGs) + i) * KCACHE2)
            .get();
    if (slmReadA == 2) {
      __builtin_IB_simd_block_write_2_local_h(
          l_addrA, __builtin_IB_simd_block_read_2_global_h(g_addrA));
    } else if (slmReadA == 4) {
      __builtin_IB_simd_block_write_4_local_h(
          l_addrA, __builtin_IB_simd_block_read_4_global_h(g_addrA));
    } else
      assert(slmReadA == 2 || slmReadA == 4);
  }
  // how much each SG will load to SLM --> has to be contiguous
  // NCACHE2*KCACHE2/(SGs*SG_SIZE) = 16
  size_t slmRead = NCACHE2 * KCACHE2 / (SGs * SG_SIZE);
  size_t sgsPerRow = (NCACHE2 * vnniFactor) / (slmRead * SG_SIZE);
  __attribute__((opencl_global)) ushort *g_addr;
  g_addr = (__attribute__((opencl_global))
            ushort *)(B +
                      (k2 * (KCACHE2 / vnniFactor) +
                       (uint)(sg.get_group_id() / sgsPerRow)) *
                          (colsB * vnniFactor) +
                      n2 * NCACHE2 * vnniFactor +
                      (sg.get_group_id() % sgsPerRow) * (slmRead * SG_SIZE));
  __attribute__((opencl_local)) ushort *l_addr;
  l_addr =
      (__attribute__((opencl_local))
       ushort *)(tileB.template get_multi_ptr<sycl::access::decorated::yes>() +
                 ((uint)(sg.get_group_id() / sgsPerRow)) * NCACHE2 *
                     vnniFactor +
                 (sg.get_group_id() % sgsPerRow) * (slmRead * SG_SIZE))
          .get();
  if (slmRead == 16) {
    __builtin_IB_simd_block_write_16_local_h(
        l_addr, __builtin_IB_simd_block_read_16_global_h(g_addr));
  } else if (slmRead == 8) {
    __builtin_IB_simd_block_write_8_local_h(
        l_addr, __builtin_IB_simd_block_read_8_global_h(g_addr));
  } else
    assert(slmRead == 8 || slmRead == 16);
}

#endif
template <unsigned int rowsA, unsigned int colsA, unsigned int rowsB,
          unsigned int colsB, unsigned int MCACHE2, unsigned int NCACHE2,
          unsigned int KCACHE2, unsigned int vnniFactor, size_t SGs,
          typename TOperand, access::address_space Space>
inline void
slm_read_write(multi_ptr<TOperand, Space, access::decorated::yes> pA,
               multi_ptr<TOperand, Space, access::decorated::yes> pB,
               local_accessor<TOperand, 2> tileA,
               local_accessor<TOperand, 2> tileB, sub_group sg, unsigned int k2,
               size_t m2, size_t n2, size_t SG_SIZE) {
  size_t slmReadA = KCACHE2 / SG_SIZE;
  for (int i = 0; i < (MCACHE2 / SGs); i++) {
    if (slmReadA == 2) {
      vec<TOperand, 2> slmvec = sg.load<2>(
          pA +
          (m2 * MCACHE2 + (sg.get_group_id()) * (MCACHE2 / SGs) + i) * colsA +
          k2 * KCACHE2);

      sg.store<2>(tileA.template get_multi_ptr<sycl::access::decorated::no>() +
                      ((sg.get_group_id()) * (MCACHE2 / SGs) + i) * KCACHE2,
                  slmvec);
    } else if (slmReadA == 4) {
      vec<TOperand, 4> slmvec = sg.load<4>(
          pA +
          (m2 * MCACHE2 + (sg.get_group_id()) * (MCACHE2 / SGs) + i) * colsA +
          k2 * KCACHE2);

      sg.store<4>(tileA.template get_multi_ptr<sycl::access::decorated::no>() +
                      ((sg.get_group_id()) * (MCACHE2 / SGs) + i) * KCACHE2,
                  slmvec);
    } else
      assert(slmReadA == 2 || slmReadA == 4);
  }
  // how much each SG will load to SLM --> has to be contiguous
  // NCACHE2*KCACHE2/(SGs*SG_SIZE) = 16
  size_t slmRead = NCACHE2 * KCACHE2 / (SGs * SG_SIZE);
  size_t sgsPerRow = (NCACHE2 * vnniFactor) / (slmRead * SG_SIZE);
  if (slmRead == 16) {
    vec<TOperand, 16> slmvecb = sg.load<16>(
        pB +
        (k2 * (KCACHE2 / vnniFactor) + (uint)(sg.get_group_id() / sgsPerRow)) *
            (colsB * vnniFactor) +
        n2 * NCACHE2 * vnniFactor +
        (sg.get_group_id() % sgsPerRow) * (slmRead * SG_SIZE));

    sg.store<16>(tileB.template get_multi_ptr<sycl::access::decorated::no>() +
                     ((uint)(sg.get_group_id() / sgsPerRow)) * NCACHE2 *
                         vnniFactor +
                     (sg.get_group_id() % sgsPerRow) * (slmRead * SG_SIZE),
                 slmvecb);
  } else if (slmRead == 8) {
    vec<TOperand, 8> slmvecb = sg.load<8>(
        pB +
        (k2 * (KCACHE2 / vnniFactor) + (uint)(sg.get_group_id() / sgsPerRow)) *
            (colsB * vnniFactor) +
        n2 * NCACHE2 * vnniFactor +
        (sg.get_group_id() % sgsPerRow) * (slmRead * SG_SIZE));

    sg.store<8>(tileB.template get_multi_ptr<sycl::access::decorated::no>() +
                    ((uint)(sg.get_group_id() / sgsPerRow)) * NCACHE2 *
                        vnniFactor +
                    (sg.get_group_id() % sgsPerRow) * (slmRead * SG_SIZE),
                slmvecb);
  } else
    assert(slmRead == 8 || slmRead == 16);
}
