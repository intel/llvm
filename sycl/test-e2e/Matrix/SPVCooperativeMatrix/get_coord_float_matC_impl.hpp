//==----------- get_coord_float_matC_impl.hpp  - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <sycl/atomic_ref.hpp>
#include <sycl/group_algorithm.hpp>

template <size_t TileRows, size_t TileCols> class add_rows;

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

template <typename T, size_t Rows, size_t Cols, size_t TileRows,
          size_t TileCols>
void matrix_sum_rows(big_matrix<T, Rows, Cols> &C, T *sum_rows) {
  buffer<T, 2> bufC((T *)C.get_data(), range<2>(Rows, Cols));
  buffer<T> sum_rows_v(sum_rows, Rows);

  queue q;
  size_t sg_size = get_sg_size<add_rows<TileRows, TileCols>>(q);
  q.submit([&](handler &cgh) {
     sycl::accessor accC{bufC, cgh, sycl::read_write};
     sycl::accessor v{sum_rows_v, cgh, sycl::read_write};

     cgh.parallel_for<add_rows<TileRows, TileCols>>(
         nd_range<2>({Rows / TileRows, Cols / TileCols * sg_size},
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
           joint_matrix<sub_group, T, use::accumulator, TileRows, TileCols>
               sub_c;

           joint_matrix_load(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TileRows) * Cols +
                   sg_starty / sg_size * TileCols,
               Cols, layout::row_major);

           T sum_local_rows[Rows] = {0};

           ext::intel::experimental::matrix::joint_matrix_apply(
               sg, sub_c, [&](T &x, size_t row, size_t col) {
                 sum_local_rows[row + global_idx * TileRows] += x;
               });
           for (int i = 0; i < Rows; i++) {
             sum_local_rows[i] =
                 reduce_over_group(sg, sum_local_rows[i], sycl::plus<>());
             // only Groups leader perform the global reduction
             if (global_idy % sg_size == 0) {
               sycl::atomic_ref<T, sycl::memory_order::relaxed,
                                sycl::memory_scope::device>
                   aref(v[i]);
               aref.fetch_add(sum_local_rows[i]);
             }
           }
         }); // parallel for
   }).wait();
}

template <typename T, size_t TM, size_t TN> void test() {
  constexpr size_t SCALE = 2;
  static constexpr size_t Rows = TM * SCALE;
  static constexpr size_t Cols = TN * SCALE;

  T sum_rows[Rows] = {0};
  T sum_rows_ref[Rows] = {0};
  T C[Rows][Cols];
  big_matrix<T, Rows, Cols> MC((T *)&C);

  matrix_rand(Rows, Cols, (T *)&C, (T)100);
  matrix_sum_rows<T, Rows, Cols, TM, TN>(MC, sum_rows);

  for (int i = 0; i < Rows; i++) {
    for (int j = 0; j < Cols; j++) {
      sum_rows_ref[i] += C[i][j];
    }
    assert(std::fabs(sum_rows_ref[i] - sum_rows[i]) <= FLOAT_EPSILON);
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
      test<float, /*TM*/ 16, /*TN*/ 16>();
      break;
    }

    if (combinations[i].nsize == 16) { // architecture::intel_gpu_pvc
      test<float, /*TM*/ 8, /*TN*/ 16>();
      break;
    }

    if (combinations[i].nsize == 8) { // architecture::intel_gpu_dg2*
      test<float, /*TM*/ 8, /*TN*/ 8>();
      break;
    }
  }
  return 0;
}
