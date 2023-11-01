//==----------- get_coord_int8_matA_impl.hpp  - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

constexpr size_t TM = 8;
constexpr size_t TK = 32;

template <typename T, size_t M, size_t K>
void sum_rows_ref(host_accessor<T, 2, access::mode::read_write> A,
                  host_accessor<int, 1, access::mode::read_write> sum_rows) {
  int sum_rows_ref[M] = {0};
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < K; j++) {
      sum_rows_ref[i] += A[i][j];
    }
    auto diff = sum_rows[i] - sum_rows_ref[i];
    assert(std::fabs(static_cast<int>(diff)) <=
           std::numeric_limits<int>::epsilon());
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

template <typename T, size_t M, size_t K>
void matrix_sum_rows(queue q, big_matrix<T, M, K> &A, nd_range<2> &r) {
  buffer<int8_t, 2> bufA(A.get_data(), range<2>(M, K));

  // size of vector is equal to number of rows in big matrix
  int sum_rows[M] = {0};
  buffer<int> sum_rows_v(sum_rows, M);
  q.submit([&](handler &cgh) {
     auto accA = bufA.get_access<access::mode::read_write>(cgh);
     auto v = sum_rows_v.get_access<access::mode::atomic>(cgh);

     cgh.parallel_for(r, [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(
                             SG_SZ)]] {
       const auto global_idx = spmd_item.get_global_id(0);
       const auto global_idy = spmd_item.get_global_id(1);
       const auto sg_startx = global_idx - spmd_item.get_local_id(0);
       const auto sg_starty = global_idy - spmd_item.get_local_id(1);

       sycl::sub_group sg = spmd_item.get_sub_group();
       joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major> sub_a;
       joint_matrix_load(sg, sub_a,
                         accA.template get_multi_ptr<access::decorated::no>() +
                             (sg_startx * TM * K) + sg_starty / SG_SZ * TK,
                         K);

       int32_t sum_local_rows[M] = {0};

       ext::intel::experimental::matrix::joint_matrix_apply(
           sg, sub_a, [&](int8_t &x, size_t row, size_t col) {
             sum_local_rows[row + global_idx * TM] += x;
           });
       for (int i = 0; i < M; ++i) {
         sum_local_rows[i] =
             reduce_over_group(sg, sum_local_rows[i], sycl::plus<>());

         // only Groups leader performs the global reduction
         if (global_idy % SG_SZ == 0)
           atomic_fetch_add(v[i], sum_local_rows[i]);
       }
     }); // parallel for
   }).wait();
  sum_rows_ref<T, M, K>(bufA.get_host_access(), sum_rows_v.get_host_access());
}

int main() {
  static constexpr size_t MATRIX_M = TM * 2;
  static constexpr size_t MATRIX_K = TK * 2;
  int8_t A[MATRIX_M][MATRIX_K];

  big_matrix<int8_t, MATRIX_M, MATRIX_K> MA((int8_t *)&A);

  size_t NDRangeM = MATRIX_M / TM;
  size_t NDRangeK = MATRIX_K / TK;
  queue q;
  nd_range<2> r({NDRangeM, NDRangeK * SG_SZ}, {1, 1 * SG_SZ});

  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_K; j++) {
      A[i][j] = i + j;
    }
  }

  matrix_sum_rows<int8_t, MATRIX_M, MATRIX_K>(q, MA, r);
  std::cout << "Passed\n";
  return 0;
}
