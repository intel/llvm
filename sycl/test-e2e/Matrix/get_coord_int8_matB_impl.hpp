//==----------- get_coord_int8_matB_impl.hpp  - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <sycl/group_algorithm.hpp>

constexpr size_t TK = 32;
constexpr size_t VF = 4;

template <typename T, size_t K, size_t N>
void sum_cols_ref(host_accessor<T, 2, access::mode::read_write> B,
                  host_accessor<int, 1, access::mode::read_write> sum_cols) {
  int sum_cols_ref[N] = {0};
  for (size_t j = 0; j < N; j++) {
    for (size_t i = 0; i < K; i++) {
      sum_cols_ref[j] += B[i][j];
    }
    auto diff = sum_cols[j] - sum_cols_ref[j];
    assert(std::fabs(static_cast<int>(diff)) <=
           std::numeric_limits<int>::epsilon());
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

template <typename T, size_t K, size_t N>
void matrix_sum_cols(queue q, big_matrix<T, K, N> &B,
                     big_matrix<T, K / VF, N * VF> &Bvnni, nd_range<2> &r) {
  buffer<int8_t, 2> bufB(B.get_data(), range<2>(K, N));
  buffer<int8_t, 2> bufBvnni(Bvnni.get_data(), range<2>(K / VF, N * VF));

  int sum_cols[N] = {0};
  buffer<int> sum_cols_v(sum_cols, N);

  q.submit([&](handler &cgh) {
     auto accB = bufBvnni.get_access<access::mode::read_write>(cgh);
     auto v = sum_cols_v.get_access<access::mode::atomic>(cgh);

     cgh.parallel_for(r, [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(
                             SG_SZ)]] {
       const auto global_idx = spmd_item.get_global_id(0);
       const auto global_idy = spmd_item.get_global_id(1);
       const auto sg_startx = global_idx - spmd_item.get_local_id(0);
       const auto sg_starty = global_idy - spmd_item.get_local_id(1);

       sycl::sub_group sg = spmd_item.get_sub_group();

       joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::ext_intel_packed>
           sub_b;

       joint_matrix_load(sg, sub_b,
                         accB.template get_multi_ptr<access::decorated::no>() +
                             (sg_startx * (TK / VF) * N * VF) +
                             sg_starty / SG_SZ * TN * VF,
                         N * VF);

       int32_t sum_local_cols[N] = {0};
       ext::intel::experimental::matrix::joint_matrix_apply(
           sg, sub_b, [&](int8_t &x, size_t row, size_t col) {
             // the coordinates returned are in the logical range [K,N]
             // If users want to retrieve the VNNIed coordinates, they can be
             // obtained using
             // colVNNI = col/VF
             // rowVNNI = row*VF
             size_t global_index = col + global_idy / SG_SZ * TN;
             sum_local_cols[global_index] += x;
           });

       for (int i = 0; i < N; i++) {
         sum_local_cols[i] =
             reduce_over_group(sg, sum_local_cols[i], sycl::plus<>());
         if (global_idy % SG_SZ == 0)
           atomic_fetch_add(v[i], sum_local_cols[i]);
       }
     }); // parallel for
   }).wait();
  sum_cols_ref<T, K, N>(bufB.get_host_access(), sum_cols_v.get_host_access());
}

int main() {
  static constexpr size_t scale = 2;
  static constexpr size_t MATRIX_K = TK * scale;
  static constexpr size_t MATRIX_N = TN * scale;

  int8_t B[MATRIX_K][MATRIX_N];
  big_matrix<int8_t, MATRIX_K, MATRIX_N> MB((int8_t *)&B);

  int8_t Bvnni[MATRIX_K / VF][MATRIX_N * VF];
  big_matrix<int8_t, MATRIX_K / VF, MATRIX_N * VF> MBvnni((int8_t *)&Bvnni);

  size_t NDRangeK = MATRIX_K / TK;
  size_t NDRangeN = MATRIX_N / TN;
  queue q;
  nd_range<2> r({NDRangeK, NDRangeN * SG_SZ}, {1, 1 * SG_SZ});

  for (int i = 0; i < MATRIX_K; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      B[i][j] = i + j;
    }
  }
  matrix_vnni<int8_t>(MATRIX_K, MATRIX_N, *B, *Bvnni, VF);
  // This test calculates sum of columns in the non VNNI B matrix
  matrix_sum_cols<int8_t, MATRIX_K, MATRIX_N>(q, MB, MBvnni, r);
  std::cout << "Passed\n";
  return 0;
}