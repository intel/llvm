//==---------- joint_matrix_float4_impl.hpp  - DPC++ joint_matrix-----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <sycl/usm.hpp>

// The only advertised fp4_e2m1 combination is {msize=8, nsize=16, ksize=32},
// so unlike the fp8 test TM is not a free parameter here.
constexpr size_t TM = 8;
constexpr size_t TN = 16;
constexpr size_t TK = 32;

// numElems is the packing factor of fp4_e2m1_x<numElems>: each storage element
// holds numElems 4-bit values. Matrix extents below are always expressed in
// logical (unpacked) elements, so offsets and strides into the packed A and B
// buffers divide by numElems.
//
// convertP selects where sycl::half is narrowed to fp4: when true, B is kept in
// memory as sycl::half and converted inside the kernel via
// joint_matrix_convert; when false, it is packed ahead of time by matrix_copy.
template <typename TA, typename TB, typename TC, size_t M, size_t N, size_t K,
          layout B_layout, unsigned int vnniFactor, unsigned int numElems,
          bool convertP>
void joint_matrix_gemm_vnni(
    sub_group sg, size_t sg_startx, size_t sg_starty, size_t sg_size,
    multi_ptr<TA, sycl::access::address_space::global_space,
              access::decorated::no>
        pA,
    multi_ptr<std::conditional_t<convertP, sycl::half, TB>,
              sycl::access::address_space::global_space, access::decorated::no>
        pB,
    multi_ptr<TC, sycl::access::address_space::global_space,
              access::decorated::no>
        pC) {
  joint_matrix<sub_group, TA, use::a, TM, TK, layout::row_major> sub_a;
  joint_matrix<sub_group, TB, use::b, TK, TN, B_layout> sub_b;
  joint_matrix<sub_group, TC, use::accumulator, TM, TN> sub_c;

  // B is addressed in packed storage elements unless it is still sycl::half.
  constexpr size_t BPack = convertP ? 1 : numElems;
  const size_t n_offset = sg_starty / sg_size * TN * vnniFactor;

  joint_matrix_load(sg, sub_c,
                    pC + (sg_startx * TM) * N + sg_starty / sg_size * TN, N,
                    layout::row_major);
  for (int k = 0; k < K; k += TK) {
    joint_matrix_load(sg, sub_a, pA + ((sg_startx * TM) * K + k) / numElems,
                      K / numElems);
    if constexpr (convertP) {
      joint_matrix<sub_group, sycl::half, use::b, TK, TN, B_layout> sub_bh;
      joint_matrix_load(sg, sub_bh, pB + (k * N + n_offset) / BPack,
                        N / BPack * vnniFactor);
      joint_matrix_convert(sg, sub_bh, sub_b);
    } else {
      joint_matrix_load(sg, sub_b, pB + (k * N + n_offset) / BPack,
                        N / BPack * vnniFactor);
    }
    joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  }
  joint_matrix_store(sg, sub_c,
                     pC + (sg_startx * TM) * N + sg_starty / sg_size * TN, N,
                     layout::row_major);
}

template <typename TA, typename TB, typename TC, unsigned int vnniFactor,
          layout B_layout, bool convertP>
class fp4matrix;

template <typename TA, typename TB, typename TC, size_t M, size_t N, size_t K,
          layout B_layout, unsigned int vnniFactor, unsigned int numElems,
          bool convertP>
void matrix_multiply(TC *C, TA *A,
                     std::conditional_t<convertP, sycl::half, TB> *B, queue q) {
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;

  auto pA = address_space_cast<sycl::access::address_space::global_space,
                               access::decorated::no>(A);
  auto pB = address_space_cast<sycl::access::address_space::global_space,
                               access::decorated::no>(B);
  auto pC = address_space_cast<sycl::access::address_space::global_space,
                               access::decorated::no>(C);
  using kernel_name = fp4matrix<TA, TB, TC, vnniFactor, B_layout, convertP>;
  size_t sg_size = get_sg_size<kernel_name>(q);
  q.submit([&](handler &cgh) {
     cgh.parallel_for<kernel_name>(
         nd_range<2>({NDRangeM, NDRangeN * sg_size}, {1, 1 * sg_size}),
         [=](nd_item<2> spmd_item)
#ifdef SG_SZ
             [[sycl::reqd_sub_group_size(SG_SZ)]]
#endif
         {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix_gemm_vnni<TA, TB, TC, M, N, K, B_layout, vnniFactor,
                                  numElems, convertP>(sg, sg_startx, sg_starty,
                                                      sg_size, pA, pB, pC);
         }); // parallel for
   }).wait();
}

// E2M1 only represents {0, 0.5, 1, 1.5, 2, 3, 4, 6} and their negations, so the
// random sycl::half input is not exactly representable. Round-tripping it
// through the 4-bit type makes the reference multiplication see exactly the
// values the device operates on. Unlike masking off fraction bits (what the fp8
// test does via matrix_truncate_fraction_bits) this needs no knowledge of the
// bit layout.
template <typename T4bit, unsigned int numElems>
void matrix_round_trip(queue q, unsigned int rows, unsigned int packedCols,
                       sycl::half *Mat, T4bit *Packed) {
  matrix_copy<sycl::half, T4bit, numElems>(q, rows, packedCols, Mat, Packed);
  matrix_copy<T4bit, sycl::half, numElems>(q, rows, packedCols, Packed, Mat);
}

template <typename TA, typename TB, typename TC, size_t M, size_t N, size_t K,
          unsigned int vnniFactor, bool convertP, unsigned int numElems>
void joint_matrix_verify(queue q) {
  sycl::half *Ah = malloc_shared<sycl::half>(M * K, q);
  sycl::half *Bh = malloc_shared<sycl::half>(K * N, q);
  TA *A = malloc_shared<TA>(M * K / numElems, q);
  TB *B = malloc_shared<TB>(K * N / numElems, q);
  TC *C = malloc_shared<TC>(M * N, q);
  TC *D = malloc_shared<TC>(M * N, q);

  matrix_rand<sycl::half>(M, K, Ah, 5);
  matrix_rand<sycl::half>(K, N, Bh, 5);
  // Snap the reference data to values E2M1 represents exactly.
  matrix_round_trip<TA, numElems>(q, M, K / numElems, Ah, A);
  matrix_round_trip<TB, numElems>(q, K, N / numElems, Bh, B);
  matrix_fill(M, N, C, (TC)1);
  matrix_fill(M, N, D, (TC)1);

  if constexpr (vnniFactor > 1) {
    // Apply VNNI on the sycl::half data, then pack it if the kernel expects
    // fp4 in memory.
    sycl::half *vnniBh = malloc_shared<sycl::half>(K * N, q);
    matrix_vnni(K, N, Bh, vnniBh, vnniFactor);
    if constexpr (convertP) {
      matrix_multiply<TA, TB, TC, M, N, K, layout::ext_intel_packed, vnniFactor,
                      numElems, convertP>(C, A, vnniBh, q);
    } else {
      TB *vnniB = malloc_shared<TB>(K * N / numElems, q);
      matrix_copy<sycl::half, TB, numElems>(q, K, N / numElems, vnniBh, vnniB);
      matrix_multiply<TA, TB, TC, M, N, K, layout::ext_intel_packed, vnniFactor,
                      numElems, convertP>(C, A, vnniB, q);
      free(vnniB, q);
    }
    free(vnniBh, q);
  } else { // row major
    if constexpr (convertP) {
      matrix_multiply<TA, TB, TC, M, N, K, layout::row_major, vnniFactor,
                      numElems, convertP>(C, A, Bh, q);
    } else {
      matrix_multiply<TA, TB, TC, M, N, K, layout::row_major, vnniFactor,
                      numElems, convertP>(C, A, B, q);
    }
  }
  matrix_multiply_ref<sycl::half, sycl::half, TC>(Ah, Bh, D, M, N, K);
  assert(matrix_compare(M, N, C, D));
  free(A, q);
  free(B, q);
  free(Ah, q);
  free(Bh, q);
  free(C, q);
  free(D, q);
}

template <typename TC, unsigned int numElems> void fp4_combinations(queue q) {
  // scale could be increased once these tests run on native hardware
  static constexpr size_t SCALE = 2;
  static constexpr size_t MATRIX_M = TM * SCALE;
  // satisfy 64B stride requirement in 2D block load
  static constexpr size_t MATRIX_N = std::max(TN * SCALE, 64ul);
  static constexpr size_t MATRIX_K = TK * SCALE;

  using fp4 = syclex::fp4_e2m1_x<numElems>;

  // vnniFactor 8 fills a 32-bit dword with 4-bit elements. joint_matrix_verify
  // folds the unpacked sycl::half data by vnniFactor and only then packs pairs
  // into bytes, so a dword ends up holding 8 consecutive k of one B column.
  // Packing before folding would instead give a dword spanning two columns and
  // four k, and vnniFactor would be 4; which of the two the hardware wants
  // cannot be established until IGC implements a 4-bit DPAS. Tracked by
  // GSD-9057.
  joint_matrix_verify<fp4, fp4, TC, MATRIX_M, MATRIX_N, MATRIX_K,
                      /*vnniFactor=*/8, /*convertP=*/false, numElems>(q);
  joint_matrix_verify<fp4, fp4, TC, MATRIX_M, MATRIX_N, MATRIX_K, 1, false,
                      numElems>(q);
  joint_matrix_verify<fp4, fp4, TC, MATRIX_M, MATRIX_N, MATRIX_K, 8, true,
                      numElems>(q);
  joint_matrix_verify<fp4, fp4, TC, MATRIX_M, MATRIX_N, MATRIX_K, 1, true,
                      numElems>(q);
}

int main() {
  sycl::queue q;
  if (!is_type_supported_by_device(q, matrix_type::fp4_e2m1)) {
    std::cout << "fp4_e2m1 type not supported on this device" << std::endl;
    return 0;
  }
  // A matrix element must pack two 4-bit values into each byte, so
  // fp4_e2m1_x<2> is the only usable packing factor; fp4_e2m1_x<1> would leave
  // the high nibble of every byte unused.
  constexpr unsigned int numElems = 2;
  fp4_combinations<float, numElems>(q);
#if 0
  // Disabled by lack of bfloat16 accumulator support in IGC, as for the fp8
  // combinations in joint_matrix_float8_impl.hpp. Tracked by Jira GSD-10112.
  fp4_combinations<bfloat16, numElems>(q);
#endif
  std::cout << "Passed\n";
  return 0;
}
