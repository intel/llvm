//==----------- get_coord_int8_matA_impl.hpp  - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <sycl/atomic_ref.hpp>
#include <sycl/group_algorithm.hpp>

template <size_t TileRows, size_t TileCols> class add_rows;

template <typename T, typename TResult, size_t Rows, size_t Cols>
void sum_rows_ref(
    host_accessor<T, 2, access::mode::read_write> A,
    host_accessor<TResult, 1, access::mode::read_write> sum_rows) {
  int sum_rows_ref[Rows] = {0};
  for (size_t i = 0; i < Rows; i++) {
    for (size_t j = 0; j < Cols; j++) {
      sum_rows_ref[i] += A[i][j];
    }
    auto diff = sum_rows[i] - sum_rows_ref[i];
    assert(std::fabs(static_cast<TResult>(diff)) <=
           std::numeric_limits<TResult>::epsilon());
  }
}

// clang-format off
/* For sub group size = 16:
wi [0,0] -> i=0, [0, 0]        wi [0,1] --> i=0, [0, 2]     wi [0,15] --> i=0, [0, 30]
            i=1, [0, 1]                     i=1, [0, 3]                   i=1, [0, 31]
            i=2, [1, 0]                     i=2, [1, 2]                   i=2, [1, 30]
            i=3, [1, 1]                     i=3, [1, 3]                   i=3, [1, 31]
            i=4, [2, 0]                     i=4, [2, 2]                   ...
            i=5, [2, 1]                     i=5, [2, 3]
            ...                             ....
            i=14,[7, 0]                     i=14, [7, 2]
            i=15,[7, 1]                     i=15, [7, 3]                  i=15, [7, 31]

Here's how the distribution of the A matrix looks like for this test case
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
<---------------------------------  SG1 --------------------------------->
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
<0> <1>  <2> <3>    <4> <5>  <6> <7>  ..... WORK ITEMS
Each work item has 16 elements <8 rows and 2 cols of the original matrix>
the data_slice holds the matrix elements in the following order:
0 0  0 0
   /
  /
1 1  1 1
   /
  /
2 2  2 2
  /
 / 
3 3  3 3 
W0 --> 0 0 1 1 2 2 3 3 .... 7 7
*/
// clang-format on

template <typename T, typename TResult, size_t Rows, size_t Cols,
          size_t TileRows, size_t TileCols>
void matrix_sum_rows(big_matrix<T, Rows, Cols> &A) {
  buffer<T, 2> bufA(A.get_data(), range<2>(Rows, Cols));

  // size of vector is equal to number of rows in big matrix
  TResult sum_rows[Rows] = {0};
  buffer<TResult> sum_rows_v(sum_rows, Rows);
  queue q;
  size_t sg_size = get_sg_size<add_rows<TileRows, TileCols>>(q);
  nd_range<2> r({Rows / TileRows, Cols / TileCols * sg_size}, {1, 1 * sg_size});
  q.submit([&](handler &cgh) {
     sycl::accessor accA{bufA, cgh, sycl::read_write};
     sycl::accessor v{sum_rows_v, cgh, sycl::read_write};

     cgh.parallel_for<add_rows<TileRows, TileCols>>(
         r, [=](nd_item<2> spmd_item)
#ifdef SG_SZ
                [[intel::reqd_sub_group_size(SG_SZ)]]
#endif
         {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sycl::sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T, use::a, TileRows, TileCols,
                        layout::row_major>
               sub_a;
           joint_matrix_load(
               sg, sub_a,
               accA.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TileRows * Cols) +
                   sg_starty / sg_size * TileCols,
               Cols);

           TResult sum_local_rows[Rows] = {0};

           ext::intel::experimental::matrix::joint_matrix_apply(
               sg, sub_a, [&](T &x, size_t row, size_t col) {
                 sum_local_rows[row + global_idx * TileRows] += x;
               });
           for (int i = 0; i < Rows; ++i) {
             sum_local_rows[i] =
                 reduce_over_group(sg, sum_local_rows[i], sycl::plus<>());

             // only Groups leader performs the global reduction
             if (global_idy % sg_size == 0) {
               sycl::atomic_ref<TResult, sycl::memory_order::relaxed,
                                sycl::memory_scope::device>
                   aref(v[i]);
               aref.fetch_add(sum_local_rows[i]);
             }
           }
         }); // parallel for
   }).wait();
  sum_rows_ref<T, TResult, Rows, Cols>(bufA.get_host_access(),
                                       sum_rows_v.get_host_access());
}

template <typename T, typename TResult, size_t TM, size_t TK> void test() {
  static constexpr size_t Rows = TM * 2;
  static constexpr size_t Cols = TK * 2;
  T A[Rows][Cols];

  big_matrix<T, Rows, Cols> MA((T *)&A);

  for (int i = 0; i < Rows; i++) {
    for (int j = 0; j < Cols; j++) {
      A[i][j] = i + j;
    }
  }

  matrix_sum_rows<T, TResult, Rows, Cols, TM, TK>(MA);
}

int main() {
  queue q;
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();

  for (unsigned int i = 0; i < combinations.size(); i++) {
    if (combinations[i].nsize == 0) { // Intel AMX
      test<int8_t, int, /*TM*/ 16, /*TK*/ 64>();
      break;
    }

    if (combinations[i].nsize == 16) { // architecture::intel_gpu_pvc
      test<int8_t, int, /*TM*/ 8, /*TK*/ 32>();
      break;
    }

    if (combinations[i].nsize == 8) { // architecture::intel_gpu_dg2*
      test<int8_t, int, /*TM*/ 8, /*TK*/ 32>();
      break;
    }
  }
  return 0;
}
