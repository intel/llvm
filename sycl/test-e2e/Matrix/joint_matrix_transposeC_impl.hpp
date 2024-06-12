//===---joint_matrix_transposeC_impl.hpp - DPC++ joint_matrix--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/usm.hpp>

template <size_t TileRows, size_t TileCols> class LS;

template <size_t TM, size_t TN, typename T1, size_t NUM_ROWS, size_t NUM_COLS>
void matrix_load_and_store(T1 *input, T1 *out_col_major, T1 *out_row_major,
                           queue q) {
  size_t M = NUM_ROWS;
  size_t N = NUM_COLS;

  static_assert((NUM_ROWS % TM) == 0);
  static_assert((NUM_COLS % TN) == 0);

  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  size_t sg_size = get_sg_size<class LS<TM, TN>>(q);

  q.submit([&](handler &cgh) {
     cgh.parallel_for<class LS<TM, TN>>(
         nd_range<2>({NDRangeM, NDRangeN * sg_size}, {1, 1 * sg_size}),
         [=](nd_item<2> spmd_item)
#ifdef SG_SZ
             [[intel::reqd_sub_group_size(SG_SZ)]]
#endif
         {
           auto p_input =
               address_space_cast<sycl::access::address_space::global_space,
                                  sycl::access::decorated::no>(input);

           auto p_out_col_major =
               address_space_cast<sycl::access::address_space::global_space,
                                  sycl::access::decorated::no>(out_col_major);
           auto p_out_row_major =
               address_space_cast<sycl::access::address_space::global_space,
                                  sycl::access::decorated::no>(out_row_major);

           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T1, use::accumulator, TM, TN> sub_matrix;

           auto row_major_offset =
               (sg_startx * TM) * N + (sg_starty / sg_size * TN);
           auto col_major_offset =
               (sg_startx * TM) + (sg_starty / sg_size * TN) * M;

           joint_matrix_load(sg, sub_matrix, p_input + col_major_offset, M,
                             layout::col_major);

           joint_matrix_store(sg, sub_matrix,
                              p_out_col_major + row_major_offset, N,
                              layout::row_major);

           joint_matrix_store(sg, sub_matrix,
                              p_out_row_major + col_major_offset, M,
                              layout::col_major);
         }); // parallel for
   }).wait();
}

template <typename T, size_t TM, size_t TN> void run_matrix_test() {
  static constexpr size_t MATRIX_M = TM * 16;
  static constexpr size_t MATRIX_N = TN * 16;

  queue q;
  T *input = malloc_shared<T>(MATRIX_M * MATRIX_N, q);
  T *out_col_major = malloc_shared<T>(MATRIX_M * MATRIX_N, q);
  T *out_row_major = malloc_shared<T>(MATRIX_M * MATRIX_N, q);
  T *ref_col_major = malloc_shared<T>(MATRIX_M * MATRIX_N, q);

  // input is column majot matrix so it is of NxM shape
  matrix_rand(MATRIX_N, MATRIX_M, input, (T)5.0);
  matrix_fill(MATRIX_M, MATRIX_N, out_col_major, (T)0);
  matrix_fill(MATRIX_N, MATRIX_M, out_row_major, (T)0);
  matrix_transpose(MATRIX_N, MATRIX_M, ref_col_major, input);

  matrix_load_and_store<TM, TN, T, MATRIX_M, MATRIX_N>(input, out_col_major,
                                                       out_row_major, q);

  // we use exact comparison as no low precision calculation is used in this
  // test
  std::cout << "compare results for: " << TM << " x " << TN << " [TM x TN]"
            << std::endl;
  bool res =
      matrix_compare<T, T, true>(MATRIX_M, MATRIX_N, out_col_major,
                                 ref_col_major) &&
      matrix_compare<T, T, true>(MATRIX_N, MATRIX_M, out_row_major, input);
  free(input, q);
  free(out_col_major, q);
  free(out_row_major, q);
  free(ref_col_major, q);
  assert(res);
}

int main() {
  queue q;
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();

  for (unsigned int i = 0; i < combinations.size(); i++) {
    if (combinations[i].nsize == 0) { // Intel AMX
      run_matrix_test<float, /*TM*/ 8, /*TN*/ 16>();
      run_matrix_test<float, /*TM*/ 7, /*TN*/ 16>();
      run_matrix_test<float, /*TM*/ 6, /*TN*/ 16>();
      run_matrix_test<float, /*TM*/ 5, /*TN*/ 16>();
      run_matrix_test<float, /*TM*/ 4, /*TN*/ 16>();
      run_matrix_test<float, /*TM*/ 3, /*TN*/ 16>();
      run_matrix_test<float, /*TM*/ 2, /*TN*/ 16>();
      run_matrix_test<float, /*TM*/ 1, /*TN*/ 16>();
      break;
    }

    if (combinations[i].nsize == 16) { // architecture::intel_gpu_pvc
      run_matrix_test<float, /*TM*/ 8, /*TN*/ 16>();
      run_matrix_test<float, /*TM*/ 7, /*TN*/ 16>();
      run_matrix_test<float, /*TM*/ 6, /*TN*/ 16>();
      run_matrix_test<float, /*TM*/ 5, /*TN*/ 16>();
      run_matrix_test<float, /*TM*/ 4, /*TN*/ 16>();
      run_matrix_test<float, /*TM*/ 3, /*TN*/ 16>();
      run_matrix_test<float, /*TM*/ 2, /*TN*/ 16>();
      run_matrix_test<float, /*TM*/ 1, /*TN*/ 16>();
      break;
    }

    if (combinations[i].nsize == 8) { // architecture::intel_gpu_dg2*
      run_matrix_test<float, /*TM*/ 8, /*TN*/ 8>();
      run_matrix_test<float, /*TM*/ 7, /*TN*/ 8>();
      run_matrix_test<float, /*TM*/ 6, /*TN*/ 8>();
      run_matrix_test<float, /*TM*/ 5, /*TN*/ 8>();
      run_matrix_test<float, /*TM*/ 4, /*TN*/ 8>();
      run_matrix_test<float, /*TM*/ 3, /*TN*/ 8>();
      run_matrix_test<float, /*TM*/ 2, /*TN*/ 8>();
      run_matrix_test<float, /*TM*/ 1, /*TN*/ 8>();
      break;
    }
  }
  return 0;
}
