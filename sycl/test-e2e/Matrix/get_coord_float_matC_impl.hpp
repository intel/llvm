//==----------- get_coord_float_matC_impl.hpp  - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
constexpr size_t TM = 8;

// clang-format off
/*
Here's how the data is distributed for sub group size = 16 on PVC
W0 --> 0 1 2 3 4 5 6 7
wi [0,0] -> i=0, [0, 0]        wi [0,1] --> i=0, [0, 1]     wi [0,15] --> i=0, [0, 15]
            i=1, [1, 0]                     i=1, [1, 1]                   i=1, [1, 15]
            i=2, [2, 0]                     i=2, [2, 1]                   ...
            ...                             ....
            i=7, [7, 0]                     i=7, [7, 1]
*/
// clang-format on

template <typename T1, size_t M, size_t N>
void matrix_sum_rows(big_matrix<T1, M, N> &C, float *sum_rows) {
  buffer<float, 2> bufC((float *)C.get_data(), range<2>(M, N));
  buffer<float> sum_rows_v(sum_rows, M);

  queue q;
  q.submit([&](handler &cgh) {
     auto accC = bufC.get_access<access::mode::read_write>(cgh);
     auto v = sum_rows_v.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for(
         nd_range<2>({M / TM, N / TN * SG_SZ}, {1, 1 * SG_SZ}),
         [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           // The submatrix API has to be accessed by all the workitems in a
           // subgroup these functions will be called once by the subgroup no
           // code divergence between the workitems
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, float, use::accumulator, TM, TN> sub_c;

           joint_matrix_load(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / SG_SZ * TN,
               N, layout::row_major);

           float sum_local_rows[M] = {0};

           ext::intel::experimental::matrix::joint_matrix_apply(
               sg, sub_c, [&](float &x, size_t row, size_t col) {
                 sum_local_rows[row + global_idx * TM] += x;
               });
           for (int i = 0; i < M; i++) {
             sum_local_rows[i] =
                 reduce_over_group(sg, sum_local_rows[i], sycl::plus<>());
             // only Groups leader perform the global reduction
             if (global_idy % SG_SZ == 0) {
               sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                sycl::memory_scope::device>
                   aref(v[i]);
               aref.fetch_add(sum_local_rows[i]);
             }
           }
         }); // parallel for
   }).wait();
}

int main() {
  constexpr size_t SCALE = 2;
  static constexpr size_t MATRIX_M = TM * SCALE;
  static constexpr size_t MATRIX_N = TN * SCALE;

  float sum_rows[MATRIX_M] = {0};
  float sum_rows_ref[MATRIX_M] = {0};
  float C[MATRIX_M][MATRIX_N];
  big_matrix<float, MATRIX_M, MATRIX_N> MC((float *)&C);

  matrix_rand(MATRIX_M, MATRIX_N, (float *)&C, (float)100);
  matrix_sum_rows(MC, sum_rows);

  bool res = true;
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      sum_rows_ref[i] += C[i][j];
    }
    if (std::fabs(sum_rows_ref[i] - sum_rows[i]) > FLOAT_EPSILON)
      res = false;
  }

  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
}
