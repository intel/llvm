//==----------- element_wise_all_ops_impl.hpp  - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
template <typename T, size_t NUM_ROWS, size_t NUM_COLS>
void assert_ops_ref(host_accessor<T, 2, access::mode::read> mat,
                    const float ref) {
  for (size_t i = 0; i < NUM_ROWS; i++)
    for (size_t j = 0; j < NUM_COLS; j++) {
      float diff;
      if constexpr (std::is_same_v<T, bfloat16>)
        diff = make_fp32(mat[i][j]) - ref;
      else
        diff = mat[i][j] - ref;
      assert(std::fabs(static_cast<float>(diff)) <
             std::numeric_limits<float>::epsilon());
    }
}

template <typename T, size_t NUM_ROWS, size_t NUM_COLS, size_t SUB_ROWS,
          size_t SUB_COLS, typename OP>
void verify_op_a(const T l, const T r, const float ref, OP op) {
  T mat[NUM_ROWS][NUM_COLS];
  big_matrix<T, NUM_ROWS, NUM_COLS> big_mat((T *)&mat);

  buffer<T, 2> bufMat(big_mat.get_data(), range<2>(NUM_ROWS, NUM_COLS));

  queue q;
  q.submit([&](handler &cgh) {
     sycl::accessor accessMat{bufMat, cgh, sycl::read_write};
     cgh.parallel_for(
         nd_range<2>({NUM_ROWS / SUB_ROWS, NUM_COLS / SUB_COLS * SG_SZ},
                     {1, 1 * SG_SZ}),
         [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T, use::a, SUB_ROWS, SUB_COLS,
                        layout::row_major>
               sub_mat;
           joint_matrix_fill(sg, sub_mat, l);
           joint_matrix_apply(sg, sub_mat, [=](T &x) { x = op(x, r); });
           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_mat,
               accessMat.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * SUB_ROWS) * NUM_COLS +
                   sg_starty / SG_SZ * SUB_COLS,
               NUM_COLS);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, NUM_ROWS, NUM_COLS>(bufMat.get_host_access(read_only), ref);
}

template <typename T, size_t NUM_ROWS, size_t NUM_COLS, size_t SUB_ROWS,
          size_t SUB_COLS, typename OP>
void verify_op_c(const T l, const T r, const float ref, OP op) {
  T mat[NUM_ROWS][NUM_COLS];
  big_matrix<T, NUM_ROWS, NUM_COLS> big_mat((T *)&mat);

  buffer<T, 2> bufMat(big_mat.get_data(), range<2>(NUM_ROWS, NUM_COLS));

  queue q;
  q.submit([&](handler &cgh) {
     sycl::accessor accessMat{bufMat, cgh, sycl::read_write};
     cgh.parallel_for(
         nd_range<2>({NUM_ROWS / SUB_ROWS, NUM_COLS / SUB_COLS * SG_SZ},
                     {1, 1 * SG_SZ}),
         [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T, use::accumulator, SUB_ROWS, SUB_COLS>
               sub_mat;
           joint_matrix_fill(sg, sub_mat, l);
           joint_matrix_apply(sg, sub_mat, [=](T &x) { x = op(x, r); });

           joint_matrix_store(
               sg, sub_mat,
               accessMat.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * SUB_ROWS) * NUM_COLS +
                   sg_starty / SG_SZ * SUB_COLS,
               NUM_COLS, layout::row_major);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, NUM_ROWS, NUM_COLS>(bufMat.get_host_access(read_only), ref);
}

template <typename T, size_t NROWS, size_t NCOLS, size_t SROWS, size_t SCOLS>
void test_ewops_a() {

  verify_op_a<T, NROWS, NCOLS, SROWS, SCOLS>(
      T(5.0), T(2.0), 7.0, [](auto l, auto r) { return l + r; });
  verify_op_a<T, NROWS, NCOLS, SROWS, SCOLS>(
      T(5.0), T(2.0), 3.0, [](auto l, auto r) { return l - r; });
  verify_op_a<T, NROWS, NCOLS, SROWS, SCOLS>(
      T(5.0), T(2.0), 10.0, [](auto l, auto r) { return l * r; });
  verify_op_a<T, NROWS, NCOLS, SROWS, SCOLS>(
      T(5.0), T(2.0), 2.5, [](auto l, auto r) { return l / r; });
  verify_op_a<T, NROWS, NCOLS, SROWS, SCOLS>(
      T(5.0), T(5.0), 5.0, [](auto l, auto r) { return l == r ? l : T(1.0); });
  verify_op_a<T, NROWS, NCOLS, SROWS, SCOLS>(
      T(5.0), T(4.0), 4.0, [](auto l, auto r) { return l == r ? l : r; });
  verify_op_a<T, NROWS, NCOLS, SROWS, SCOLS>(
      T(5.0), T(5.0), 1.0, [](auto l, auto r) { return l != r ? l : T(1.0); });
  verify_op_a<T, NROWS, NCOLS, SROWS, SCOLS>(
      T(5.0), T(2.0), 3.0,
      [](auto l, auto r) { return l > r ? T(3.0) : T(2.0); });
  verify_op_a<T, NROWS, NCOLS, SROWS, SCOLS>(
      T(5.0), T(2.0), 2.0,
      [](auto l, auto r) { return l < r ? T(3.0) : T(2.0); });
  verify_op_a<T, NROWS, NCOLS, SROWS, SCOLS>(
      T(5.0), T(2.0), 3.0,
      [](auto l, auto r) { return l >= r ? T(3.0) : T(2.0); });
  verify_op_a<T, NROWS, NCOLS, SROWS, SCOLS>(
      T(5.0), T(2.0), 2.0,
      [](auto l, auto r) { return l <= r ? T(3.0) : T(2.0); });
}

template <typename T, size_t NROWS, size_t NCOLS, size_t SROWS, size_t SCOLS>
void test_ewops_c() {

  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS>(
      T(5.0), T(2.0), 7.0, [](auto l, auto r) { return l + r; });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS>(
      T(5.0), T(2.0), 3.0, [](auto l, auto r) { return l - r; });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS>(
      T(5.0), T(2.0), 10.0, [](auto l, auto r) { return l * r; });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS>(
      T(5.0), T(2.0), 2.5, [](auto l, auto r) { return l / r; });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS>(
      T(5.0), T(5.0), 5.0, [](auto l, auto r) { return l == r ? l : T(1.0); });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS>(
      T(5.0), T(4.0), 4.0, [](auto l, auto r) { return l == r ? l : r; });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS>(
      T(5.0), T(5.0), 1.0, [](auto l, auto r) { return l != r ? l : T(1.0); });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS>(
      T(5.0), T(2.0), 3.0,
      [](auto l, auto r) { return l > r ? T(3.0) : T(2.0); });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS>(
      T(5.0), T(2.0), 2.0,
      [](auto l, auto r) { return l < r ? T(3.0) : T(2.0); });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS>(
      T(5.0), T(2.0), 3.0,
      [](auto l, auto r) { return l >= r ? T(3.0) : T(2.0); });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS>(
      T(5.0), T(2.0), 2.0,
      [](auto l, auto r) { return l <= r ? T(3.0) : T(2.0); });
}

int main() {
  static constexpr size_t TM = 8;
  static constexpr size_t TK = 16;

  static constexpr size_t MATRIX_M = TM * 2;
  static constexpr size_t MATRIX_N = TN * 2;
  static constexpr size_t MATRIX_K = TK * 2;
  queue q;
  if (is_type_supported_by_device(q, matrix_type::bf16)) {
    test_ewops_a<bfloat16, MATRIX_M, MATRIX_K, TM, TK>();
    test_ewops_c<float, MATRIX_M, MATRIX_N, TM, TN>();
  }

  return 0;
}
