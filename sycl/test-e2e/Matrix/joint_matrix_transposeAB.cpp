//===---joint_matrix_transposeAB.cpp - DPC++ joint_matrix--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: target-nvidia, target-amd
// UNSUPPORTED-INTENDED: aspect-ext_intel_matrix isn't currently supported for
// other triples

// REQUIRES: aspect-ext_intel_matrix

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// SG size = 32 is not currently supported for SYCL Joint Matrix by IGC on DG2
// RUN: %if !arch-intel_gpu_dg2 %{ %{build} -o %t_sg32.out -DSG_SZ=32 %}
// RUN: %if !arch-intel_gpu_dg2 %{ %{run} %t_sg32.out %}

// XFAIL: run-mode && gpu-intel-dg2
// XFAIL-TRACKER: GSD-5768

// XFAIL: cpu
// XFAIL-TRACKER: CMPLRLLVM-52693

#include "common.hpp"
#include <sycl/usm.hpp>

template <typename T, size_t TileRows, size_t TileCols, use Use> class MT;

template <size_t TR, size_t TC, typename T, size_t NR, size_t NC, use Use>
void matrix_transpose(T *in, T *out, queue q) {
  static_assert((NR % TR) == 0);
  static_assert((NC % TC) == 0);
  size_t sg_size = get_sg_size<class MT<T, TR, TC, Use>>(q);
  std::cout << "SG size " << sg_size << " ";

  q.submit([&](handler &cgh) {
     cgh.parallel_for<class MT<T, TR, TC, Use>>(
         nd_range<2>({NR / TR, NC / TC * sg_size}, {1, 1 * sg_size}),
         [=](nd_item<2> spmd_item)
#ifdef SG_SZ
             [[sycl::reqd_sub_group_size(SG_SZ)]]
#endif
         {
           auto in_ptr =
               address_space_cast<sycl::access::address_space::global_space,
                                  sycl::access::decorated::no>(in);
           auto out_ptr =
               address_space_cast<sycl::access::address_space::global_space,
                                  sycl::access::decorated::no>(out);

           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T, Use, TR, TC, layout::row_major>
               matrix_row_major;
           joint_matrix<sub_group, T, Use, TR, TC, layout::col_major>
               matrix_col_major;

           auto row_major_offset =
               (sg_startx * TR) * NC + sg_starty / sg_size * TC;
           auto col_major_offset =
               (sg_startx * TR) + (sg_starty / sg_size * TC) * NR;

           joint_matrix_load(sg, matrix_row_major, in_ptr + row_major_offset,
                             NC);
           joint_matrix_copy(sg, matrix_row_major, matrix_col_major);
           ext::intel::experimental::matrix::joint_matrix_store(
               sg, matrix_col_major, out_ptr + col_major_offset, NR);
         }); // parallel for
   }).wait();
}

template <typename T, size_t TR, size_t TC, use Use> void test() {
  std::cout << "Test " << TR << " x " << TC << " ";
  static constexpr size_t SCALE = 2;
  static constexpr size_t MATRIX_R = TR * SCALE;
  static constexpr size_t MATRIX_C = TC * SCALE;

  queue q;
  T *in = malloc_shared<T>(MATRIX_R * MATRIX_C, q);
  T *col_major = malloc_shared<T>(MATRIX_C * MATRIX_R, q);
  T *ref_col_major = malloc_shared<T>(MATRIX_C * MATRIX_R, q);

  matrix_rand(MATRIX_R, MATRIX_C, in, (T)5);
  matrix_transpose<TR, TC, T, MATRIX_R, MATRIX_C, Use>(in, col_major, q);
  matrix_transpose(MATRIX_R, MATRIX_C, ref_col_major, in);
  assert((matrix_compare<T, T, true>(MATRIX_C, MATRIX_R, col_major,
                                     ref_col_major)));
  std::cout << "PASSED\n";

  free(in, q);
  free(col_major, q);
  free(ref_col_major, q);
}

int main() {
  queue q;
  std::vector<combination> combinations =
      q.get_device().get_info<syclex::info::device::matrix_combinations>();
  bool bf16_run = false;
  bool half_run = false;
  bool int8_run = false;

  for (auto &combination : combinations) {
    if (!bf16_run && combination.atype == matrix_type::bf16) {
      std::cout << "bf16:\n";
      test<bfloat16, 8, 16, use::a>();
      test<bfloat16, 16, 16, use::b>();
#ifdef MORE_SHAPES
      test<bfloat16, 1, 16, use::a>();
      test<bfloat16, 1, 32, use::a>();
      test<bfloat16, 16, 16, use::a>();
      test<bfloat16, 32, 16, use::a>();
      test<bfloat16, 32, 32, use::a>();
      test<bfloat16, 16, 64, use::b>();
      test<bfloat16, 32, 64, use::b>();
#endif
      bf16_run = true;
    }

    if (!half_run && combination.atype == matrix_type::fp16) {
      std::cout << "half:\n";
      test<half, 8, 16, use::a>();
      test<half, 16, 16, use::b>();
#ifdef MORE_SHAPES
      test<half, 1, 16, use::a>();
      test<half, 1, 32, use::a>();
      test<half, 16, 16, use::a>();
      test<half, 32, 16, use::a>();
      test<half, 32, 32, use::a>();
      test<half, 16, 64, use::b>();
      test<half, 32, 64, use::b>();
#endif
      half_run = true;
    }

    if (!int8_run && combination.atype == matrix_type::sint8) {
      std::cout << "int8:\n";
      test<int8_t, 8, 32, use::a>();
      test<int8_t, 32, 16, use::b>();
      int8_run = true;
    }

    if (bf16_run && half_run && int8_run)
      break;
  }

  return 0;
}
