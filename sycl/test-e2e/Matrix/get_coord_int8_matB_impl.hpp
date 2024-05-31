//==----------- get_coord_int8_matB_impl.hpp  - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <sycl/atomic_ref.hpp>
#include <sycl/group_algorithm.hpp>

template <size_t TileRows, size_t TileCols> class add_cols;

template <typename T, typename TResult, size_t Rows, size_t Cols>
void sum_cols_ref(
    host_accessor<T, 2, access::mode::read_write> B,
    host_accessor<TResult, 1, access::mode::read_write> sum_cols) {
  TResult sum_cols_ref[Cols] = {0};
  for (size_t j = 0; j < Cols; j++) {
    for (size_t i = 0; i < Rows; i++) {
      sum_cols_ref[j] += B[i][j];
    }
    auto diff = sum_cols[j] - sum_cols_ref[j];
    assert(std::fabs(static_cast<TResult>(diff)) <=
           std::numeric_limits<TResult>::epsilon());
  }
}

// clang-format off
/* 
    Here is a demonstration of how matrix B will be divided across
    work items for this test case for sub group size = 16 on PVC.
    <    ---------------    128    ---------------------------------->
    x x x x x x x x x x x x x x x x       ..........    x x x x x x   ^
    x x x x x x x x x x x x x x x x       ..........    x x x x x x  16
    x x x x x x x x x x x x x x x x       ..........    x x x x x x   |
    .....                                                             |
    x x x x x x x x x x x x x x x x       ..........    x x x x x x   |
    x x x x x x x x x x x x x x x x       ..........    x x x x x x   v
    
    ---------------    64    ---------------->
    x x x x   x x    ..........    x x  x x x x   ^
    x x x x   x x    ..........    x x  x x x x   8
    x x x x   x x    ..........    x x  x x x x   |  <-- part of (VNNI-ed) 
    .....                                         |   original matrix each SG
    x x x x   x x    ..........    x x  x x x x   |   holds
    x x x x   x x    ..........    x x  x x x x   v
    < WI0 >                            < WI15 >
    <--------    16    ------------->
    x x x     ..........    x x x   ^
    x x x     ..........    x x x   |
    x x x     ..........    x x x   | <-- part of (non-VNNI-ed) original matrix
    .....                           |           each SG holds
    x x x     ..........    x x x   |
    x x x     ..........    x x x   |
    x x x     ..........    x x x  32
    x x x     ..........    x x x   |
    x x x     ..........    x x x   |
    x x x     ..........    x x x   |
    x x x     ..........    x x x   |
    x x x     ..........    x x x   |
    x x x     ..........    x x x   v
    If we divide the above matrix across 16 (SG_SZ) work items,
    each WI will hold 32 elements.  And these 32 elements will be
    8x4 chunks as shown in the VNNI-ed matrix figure. 

The total distribution among the WIs in ALL the sub-groups is as follows:
This is useful to figure out how the global index is to be calculated

W0 --> 0 0 0 0   1 1 1 1 ...   7 7 7 7 --> total 32 elements
wi [0,0] --> i=0, [0, 0]        wi [0,1] --> i=0, [0, 4]     wi [0,15] --> i=0, [0, 60] | wi [0,16] --> i=0, [0, 64]
            i=1, [0, 1]                     i=1, [0, 5]                   i=1, [0, 61]  |               i=1, [0, 65]
            i=2, [0, 2]                     i=2, [0, 6]                   i=2, [0, 62]  |               i=2, [0, 66]
            i=3, [0, 3]                     i=3, [0, 7]                   i=3, [0, 63]  |               i=3, [0, 67]              
            i=4, [1, 0]                     i=4, [1, 4]                   i=4, [1, 60]  |               ....
            i=5, [1, 1]                     i=5, [1, 5]                   i=5, [1, 61]  |
            i=6, [1, 2]                     i=6, [1, 6]                   i=6, [1, 62]  |
            i=7, [1, 3]                     i=7, [1, 7]                   i=7, [1, 63]  |
            ...                             ...                           ....          |
            i=28,[7, 0]                     i=28,[7, 4]                   i=28,[7, 60]  |               i=28, [7, 124]
            i=29,[7, 1]                     i=29,[7, 5]                   i=29,[7, 61]  |               i=29, [7, 125]
            i=30,[7, 2]                     i=30,[7, 6]                   i=30,[7, 62]  |               i=30, [7, 126]
            i=31,[7, 3]                     i=31,[7, 7]                   i=31,[7, 63]  |               i=31, [7, 127]
---------------------------------------------------------------------------------------- ---------------------------
wi [1,0] -->    i=0, [8, 0]
                i=1, [8, 1]
                i=2, [8, 2]
                i=3, [8, 2]
                ...
                i=28, [15, 0]
                i=29, [15, 1]
                i=30, [15, 2]
                i=31, [15, 3]
*/

// clang-format on

template <typename T, typename TResult, size_t Rows, size_t Cols,
          size_t TileRows, size_t TileCols, size_t VNNI>
void matrix_sum_cols(big_matrix<T, Rows, Cols> &B,
                     big_matrix<T, Rows / VNNI, Cols * VNNI> &Bvnni) {
  buffer<T, 2> bufB(B.get_data(), range<2>(Rows, Cols));
  buffer<T, 2> bufBvnni(Bvnni.get_data(), range<2>(Rows / VNNI, Cols * VNNI));

  TResult sum_cols[Cols] = {0};
  buffer<TResult> sum_cols_v(sum_cols, Cols);

  size_t NDRangeK = Rows / TileRows;
  size_t NDRangeN = Cols / TileCols;
  queue q;
  size_t sg_size = get_sg_size<add_cols<TileRows, TileCols>>(q);
  nd_range<2> r({NDRangeK, NDRangeN * sg_size}, {1, 1 * sg_size});

  q.submit([&](handler &cgh) {
     sycl::accessor accB{bufBvnni, cgh, sycl::read_write};
     sycl::accessor v{sum_cols_v, cgh, sycl::read_write};

     cgh.parallel_for<add_cols<TileRows, TileCols>>(
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

           joint_matrix<sub_group, T, use::b, TileRows, TileCols,
                        layout::ext_intel_packed>
               sub_b;

           joint_matrix_load(
               sg, sub_b,
               accB.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * (TileRows / VNNI) * Cols * VNNI) +
                   sg_starty / sg_size * TileCols * VNNI,
               Cols * VNNI);

           TResult sum_local_cols[Cols] = {0};
           ext::intel::experimental::matrix::joint_matrix_apply(
               sg, sub_b, [&](T &x, size_t row, size_t col) {
                 // the coordinates returned are in the logical range
                 // [Rows,Cols] If users want to retrieve the VNNIed
                 // coordinates, they can be obtained using colVNNI = col/VNNI
                 // rowVNNI = row*VNNI
                 size_t global_index = col + global_idy / sg_size * TileCols;
                 sum_local_cols[global_index] += x;
               });

           for (int i = 0; i < Cols; i++) {
             sum_local_cols[i] =
                 reduce_over_group(sg, sum_local_cols[i], sycl::plus<>());
             if (global_idy % sg_size == 0) {
               sycl::atomic_ref<TResult, sycl::memory_order::relaxed,
                                sycl::memory_scope::device>
                   aref(v[i]);
               aref.fetch_add(sum_local_cols[i]);
             }
           }
         }); // parallel for
   }).wait();
  sum_cols_ref<T, TResult, Rows, Cols>(bufB.get_host_access(),
                                       sum_cols_v.get_host_access());
}

template <typename T, typename TResult, size_t VNNI, size_t TK, size_t TN>
void test() {
  static constexpr size_t scale = 2;
  static constexpr size_t MATRIX_K = TK * scale;
  static constexpr size_t MATRIX_N = TN * scale;

  T B[MATRIX_K][MATRIX_N];
  big_matrix<T, MATRIX_K, MATRIX_N> MB((T *)&B);

  T Bvnni[MATRIX_K / VNNI][MATRIX_N * VNNI];
  big_matrix<T, MATRIX_K / VNNI, MATRIX_N * VNNI> MBvnni((T *)&Bvnni);

  for (int i = 0; i < MATRIX_K; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      B[i][j] = i + j;
    }
  }
  matrix_vnni<T>(MATRIX_K, MATRIX_N, *B, *Bvnni, VNNI);
  // This test calculates sum of columns in the non VNNI B matrix
  matrix_sum_cols<T, TResult, MATRIX_K, MATRIX_N, TK, TN, VNNI>(MB, MBvnni);
}

int main() {
  queue q;
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();

  for (unsigned int i = 0; i < combinations.size(); i++) {
    if (combinations[i].nsize == 0) { // Intel AMX
      test<int8_t, int32_t, 4, /*TK*/ 64, /*TN*/ 16>();
      break;
    }

    if (combinations[i].nsize == 16) { // architecture::intel_gpu_pvc
      test<int8_t, int32_t, 4, /*TK*/ 32, /*TN*/ 16>();
      break;
    }

    if (combinations[i].nsize == 8) { // architecture::intel_gpu_dg2*
      test<int8_t, int32_t, 4, /*TK*/ 32, /*TN*/ 8>();
      break;
    }
  }
  return 0;
}