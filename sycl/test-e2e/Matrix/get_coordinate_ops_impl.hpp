//==----------- get_coordinate_ops_impl.hpp  - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <sycl/atomic_ref.hpp>
#include <sycl/group_algorithm.hpp>

template <typename T, size_t Rows, size_t Cols, layout Layout, use Use>
class matrix_process;

template <typename T, typename TResult, size_t NUM_ROWS, size_t NUM_COLS,
          size_t SROWS, size_t SCOLS, use Use, layout Layout, size_t VF>
void matrix_sum(big_matrix<T, NUM_ROWS / VF, NUM_COLS * VF> &M,
                TResult *sum_rows, TResult *sum_cols) {
  buffer<T, 2> buf((T *)M.get_data(), range<2>(NUM_ROWS / VF, NUM_COLS * VF));
  buffer<TResult> sum_rows_v(sum_rows, NUM_ROWS);
  buffer<TResult> sum_cols_v(sum_cols, NUM_COLS);

  queue q;
  size_t sg_size =
      get_sg_size<matrix_process<T, NUM_ROWS, NUM_COLS, Layout, Use>>(q);
  q.submit([&](handler &cgh) {
     sycl::accessor acc{buf, cgh, sycl::read_write};
     sycl::accessor v_rows{sum_rows_v, cgh, sycl::read_write};
     sycl::accessor v_cols{sum_cols_v, cgh, sycl::read_write};

     cgh.parallel_for<matrix_process<T, NUM_ROWS, NUM_COLS, Layout, Use>>(
         nd_range<2>({NUM_ROWS / SROWS, NUM_COLS / SCOLS * sg_size},
                     {1, 1 * sg_size}),
         [=](nd_item<2> spmd_item)
#ifdef SG_SZ
             [[intel::reqd_sub_group_size(SG_SZ)]]
#endif
         {
           // The submatrix API has to be accessed by all the workitems in a
           // subgroup these functions will be called once by the subgroup no
           // code divergence between the workitems
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();

           TResult sum_local_rows[NUM_ROWS] = {0};
           TResult sum_local_cols[NUM_COLS] = {0};

           if (Use == use::accumulator) {
             joint_matrix<sub_group, T, use::accumulator, SROWS, SCOLS,
                          layout::dynamic>
                 sub;

             joint_matrix_load(
                 sg, sub,
                 acc.template get_multi_ptr<access::decorated::no>() +
                     (sg_startx * SROWS * NUM_COLS) +
                     sg_starty / sg_size * SCOLS,
                 NUM_COLS, Layout);

             ext::intel::experimental::matrix::joint_matrix_apply(
                 sg, sub, [&](T &x, size_t row, size_t col) {
                   sum_local_rows[row + global_idx * SROWS] += x;
                   sum_local_cols[col + global_idy / sg_size * SCOLS] += x;
                 });

           } else {
             joint_matrix<sub_group, T, Use, SROWS, SCOLS, Layout> sub;

             joint_matrix_load(
                 sg, sub,
                 acc.template get_multi_ptr<access::decorated::no>() +
                     (sg_startx * (SROWS / VF) * NUM_COLS * VF) +
                     sg_starty / sg_size * SCOLS * VF,
                 NUM_COLS * VF);

             ext::intel::experimental::matrix::joint_matrix_apply(
                 sg, sub, [&](T &x, size_t row, size_t col) {
                   sum_local_rows[row + global_idx * SROWS] += x;
                   sum_local_cols[col + global_idy / sg_size * SCOLS] += x;
                 });
           }

           for (int i = 0; i < NUM_ROWS; i++) {
             sum_local_rows[i] =
                 reduce_over_group(sg, sum_local_rows[i], sycl::plus<>());
             // only Groups leader perform the global reduction
             if (global_idy % sg_size == 0) {
               sycl::atomic_ref<TResult, sycl::memory_order::relaxed,
                                sycl::memory_scope::device>
                   aref(v_rows[i]);
               aref.fetch_add(sum_local_rows[i]);
             }
           }

           for (int i = 0; i < NUM_COLS; i++) {
             sum_local_cols[i] =
                 reduce_over_group(sg, sum_local_cols[i], sycl::plus<>());
             // only Groups leader perform the global reduction
             if (global_idy % sg_size == 0) {
               sycl::atomic_ref<TResult, sycl::memory_order::relaxed,
                                sycl::memory_scope::device>
                   aref(v_cols[i]);
               aref.fetch_add(sum_local_cols[i]);
             }
           }
         }); // parallel for
   }).wait();
}

template <typename T, typename TResult, size_t SROWS, size_t SCOLS, use Use,
          layout Layout, size_t VF>
void test_get_coord_op() {
  constexpr size_t SCALE = 2;
  static constexpr size_t Rows = SROWS * SCALE;
  static constexpr size_t Cols = SCOLS * SCALE;

  T M[Rows][Cols];
  T Mvnni[Rows / VF][Cols * VF];
  TResult sum_rows[Rows] = {0};
  TResult sum_rows_ref[Rows] = {0};
  TResult sum_cols[Cols] = {0};
  TResult sum_cols_ref[Cols] = {0};

  for (int i = 0; i < Rows; i++) {
    for (int j = 0; j < Cols; j++) {
      M[i][j] = i + j;
    }
  }

  matrix_vnni<T>(Rows, Cols, *M, *Mvnni, VF);
  big_matrix<T, Rows / VF, Cols * VF> MM((T *)&Mvnni);

  matrix_sum<T, TResult, Rows, Cols, SROWS, SCOLS, Use, Layout, VF>(
      MM, sum_rows, sum_cols);

  // This condition check can be removed once the IGC PR resolving the Matrix B row 
  // coordinate bug is pull downed to the driver.
  if (Use != use::b) {
    for (int i = 0; i < Rows; i++) {
      for (int j = 0; j < Cols; j++) {
        sum_rows_ref[i] += (int)M[i][j];
      }
      assert(std::fabs(sum_rows_ref[i] - sum_rows[i]) <= FLOAT_EPSILON);
    }
  }

  for (int j = 0; j < Cols; j++) {
    for (int i = 0; i < Rows; i++) {
      sum_cols_ref[j] += (int)M[i][j];
    }
    assert(std::fabs(sum_cols_ref[j] - sum_cols[j]) <= FLOAT_EPSILON);
  }
}

int main() {
  queue q;
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();

  for (unsigned int i = 0; i < combinations.size(); i++) {
    if (combinations[i].nsize == 0) { // Intel AMX
      test_get_coord_op<bfloat16, float, /*TM*/ 16, /*TK*/ 32, use::a,
                        layout::row_major, 1>();
      test_get_coord_op<int8_t, int, /*TM*/ 16, /*TK*/ 64, use::a,
                        layout::row_major, 1>();
      test_get_coord_op<bfloat16, float, /*TK*/ 16, /*TN*/ 16, use::b,
                        layout::row_major, 1>();
      test_get_coord_op<int8_t, int32_t, /*TK*/ 64, /*TN*/ 16, use::b,
                        layout::row_major, 1>();
      test_get_coord_op<bfloat16, float, /*TK*/ 16, /*TN*/ 16, use::b,
                        layout::ext_intel_packed, 2>();
      test_get_coord_op<int8_t, int32_t, /*TK*/ 64, /*TN*/ 16, use::b,
                        layout::ext_intel_packed, 4>();
      test_get_coord_op<float, float, /*TM*/ 16, /*TN*/ 16, use::accumulator,
                        layout::row_major, 1>();
      test_get_coord_op<int32_t, int32_t, /*TM*/ 16, /*TN*/ 16,
                        use::accumulator, layout::row_major, 1>();
      break;
    }

    if (combinations[i].nsize == 16) { // architecture::intel_gpu_pvc
      test_get_coord_op<bfloat16, float, /*TM*/ 8, /*TK*/ 16, use::a,
                        layout::row_major, 1>();
      test_get_coord_op<int8_t, int, /*TM*/ 8, /*TK*/ 32, use::a,
                        layout::row_major, 1>();
      test_get_coord_op<bfloat16, float, /*TK*/ 16, /*TN*/ 16, use::b,
                        layout::ext_intel_packed, 2>();
      test_get_coord_op<int8_t, int32_t, /*TK*/ 32, /*TN*/ 16, use::b,
                        layout::ext_intel_packed, 4>();
      test_get_coord_op<float, float, /*TM*/ 8, /*TN*/ 16, use::accumulator,
                        layout::row_major, 1>();
      test_get_coord_op<int32_t, int32_t, /*TM*/ 8, /*TN*/ 16, use::accumulator,
                        layout::row_major, 1>();
      // This combination is not currently supported for sub group size = 32 in
      // IGC
#if (!defined(SG_SZ) || SG_SZ != 32)
      test_get_coord_op<bfloat16, float, /*TK*/ 16, /*TN*/ 16, use::b,
                        layout::row_major, 1>();
      test_get_coord_op<int8_t, int32_t, /*TK*/ 32, /*TN*/ 16, use::b,
                        layout::row_major, 1>();
#endif
      break;
    }

    if (combinations[i].nsize == 8) { // architecture::intel_gpu_dg2*
      test_get_coord_op<bfloat16, float, /*TM*/ 8, /*TK*/ 16, use::a,
                        layout::row_major, 1>();
      test_get_coord_op<int8_t, int, /*TM*/ 8, /*TK*/ 32, use::a,
                        layout::row_major, 1>();
      test_get_coord_op<bfloat16, float, /*TK*/ 16, /*TN*/ 8, use::b,
                        layout::row_major, 1>();
      test_get_coord_op<int8_t, int32_t, /*TK*/ 32, /*TN*/ 8, use::b,
                        layout::row_major, 1>();
      test_get_coord_op<bfloat16, float, /*TK*/ 16, /*TN*/ 8, use::b,
                        layout::ext_intel_packed, 2>();
      test_get_coord_op<int8_t, int32_t, /*TK*/ 32, /*TN*/ 8, use::b,
                        layout::ext_intel_packed, 4>();
      test_get_coord_op<float, float, /*TM*/ 8, /*TN*/ 8, use::accumulator,
                        layout::row_major, 1>();
      test_get_coord_op<int32_t, int32_t, /*TM*/ 8, /*TN*/ 8, use::accumulator,
                        layout::row_major, 1>();
      break;
    }
  }
  return 0;
}
