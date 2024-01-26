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
          size_t SUB_COLS, class kernel_name, typename OP>
void verify_op_a(const T l, const T r, const float ref, OP op) {
  T mat[NUM_ROWS][NUM_COLS];
  big_matrix<T, NUM_ROWS, NUM_COLS> big_mat((T *)&mat);

  buffer<T, 2> bufMat(big_mat.get_data(), range<2>(NUM_ROWS, NUM_COLS));

  queue q;
  size_t sg_size = get_sg_size<kernel_name>(q);
  q.submit([&](handler &cgh) {
     sycl::accessor accessMat{bufMat, cgh, sycl::read_write};
     cgh.parallel_for<kernel_name>(
         nd_range<2>({NUM_ROWS / SUB_ROWS, NUM_COLS / SUB_COLS * sg_size},
                     {1, 1 * sg_size}),
         [=](nd_item<2> spmd_item)
#ifdef SG_SZ
             [[intel::reqd_sub_group_size(SG_SZ)]]
#endif
         {
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
                   sg_starty / sg_size * SUB_COLS,
               NUM_COLS);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, NUM_ROWS, NUM_COLS>(bufMat.get_host_access(read_only), ref);
}

template <typename T, size_t NUM_ROWS, size_t NUM_COLS, size_t SUB_ROWS,
          size_t SUB_COLS, class kernel_name, typename OP>
void verify_op_c(const T l, const T r, const float ref, OP op) {
  T mat[NUM_ROWS][NUM_COLS];
  big_matrix<T, NUM_ROWS, NUM_COLS> big_mat((T *)&mat);

  buffer<T, 2> bufMat(big_mat.get_data(), range<2>(NUM_ROWS, NUM_COLS));
  queue q;
  size_t sg_size = get_sg_size<kernel_name>(q);
  q.submit([&](handler &cgh) {
     sycl::accessor accessMat{bufMat, cgh, sycl::read_write};
     cgh.parallel_for<kernel_name>(
         nd_range<2>({NUM_ROWS / SUB_ROWS, NUM_COLS / SUB_COLS * sg_size},
                     {1, 1 * sg_size}),
         [=](nd_item<2> spmd_item)
#ifdef SG_SZ
             [[intel::reqd_sub_group_size(SG_SZ)]]
#endif
         {
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
                   sg_starty / sg_size * SUB_COLS,
               NUM_COLS, layout::row_major);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, NUM_ROWS, NUM_COLS>(bufMat.get_host_access(read_only), ref);
}

template <typename T, size_t NROWS, size_t NCOLS, size_t SROWS, size_t SCOLS>
void test_ewops_a() {

  verify_op_a<T, NROWS, NCOLS, SROWS, SCOLS, class a_add>(
      T(5.0), T(2.0), 7.0, [](auto l, auto r) { return l + r; });
  verify_op_a<T, NROWS, NCOLS, SROWS, SCOLS, class a_sub>(
      T(5.0), T(2.0), 3.0, [](auto l, auto r) { return l - r; });
  verify_op_a<T, NROWS, NCOLS, SROWS, SCOLS, class a_mul>(
      T(5.0), T(2.0), 10.0, [](auto l, auto r) { return l * r; });
  verify_op_a<T, NROWS, NCOLS, SROWS, SCOLS, class a_div>(
      T(5.0), T(2.0), 2.5, [](auto l, auto r) { return l / r; });
  verify_op_a<T, NROWS, NCOLS, SROWS, SCOLS, class a_logical>(
      T(5.0), T(5.0), 5.0, [](auto l, auto r) { return l == r ? l : T(1.0); });
  verify_op_a<T, NROWS, NCOLS, SROWS, SCOLS, class a_eq>(
      T(5.0), T(4.0), 4.0, [](auto l, auto r) { return l == r ? l : r; });
  verify_op_a<T, NROWS, NCOLS, SROWS, SCOLS, class a_ne>(
      T(5.0), T(5.0), 1.0, [](auto l, auto r) { return l != r ? l : T(1.0); });
  verify_op_a<T, NROWS, NCOLS, SROWS, SCOLS, class a_gt>(
      T(5.0), T(2.0), 3.0,
      [](auto l, auto r) { return l > r ? T(3.0) : T(2.0); });
  verify_op_a<T, NROWS, NCOLS, SROWS, SCOLS, class a_lt>(
      T(5.0), T(2.0), 2.0,
      [](auto l, auto r) { return l < r ? T(3.0) : T(2.0); });
  verify_op_a<T, NROWS, NCOLS, SROWS, SCOLS, class a_ge>(
      T(5.0), T(2.0), 3.0,
      [](auto l, auto r) { return l >= r ? T(3.0) : T(2.0); });
  verify_op_a<T, NROWS, NCOLS, SROWS, SCOLS, class a_le>(
      T(5.0), T(2.0), 2.0,
      [](auto l, auto r) { return l <= r ? T(3.0) : T(2.0); });
}
// Avoid same kernel name for different Sg sizes
template <size_t COLS, class name> class ewops_c {};
template <typename T, size_t NROWS, size_t NCOLS, size_t SROWS, size_t SCOLS>
void test_ewops_c() {

  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS, ewops_c<SCOLS, class c_add>>(
      T(5.0), T(2.0), 7.0, [](auto l, auto r) { return l + r; });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS, ewops_c<SCOLS, class c_sub>>(
      T(5.0), T(2.0), 3.0, [](auto l, auto r) { return l - r; });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS, ewops_c<SCOLS, class c_mul>>(
      T(5.0), T(2.0), 10.0, [](auto l, auto r) { return l * r; });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS, ewops_c<SCOLS, class c_div>>(
      T(5.0), T(2.0), 2.5, [](auto l, auto r) { return l / r; });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS, ewops_c<SCOLS, class c_logical>>(
      T(5.0), T(5.0), 5.0, [](auto l, auto r) { return l == r ? l : T(1.0); });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS, ewops_c<SCOLS, class c_eq>>(
      T(5.0), T(4.0), 4.0, [](auto l, auto r) { return l == r ? l : r; });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS, ewops_c<SCOLS, class c_ne>>(
      T(5.0), T(5.0), 1.0, [](auto l, auto r) { return l != r ? l : T(1.0); });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS, ewops_c<SCOLS, class c_gt>>(
      T(5.0), T(2.0), 3.0,
      [](auto l, auto r) { return l > r ? T(3.0) : T(2.0); });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS, ewops_c<SCOLS, class c_lt>>(
      T(5.0), T(2.0), 2.0,
      [](auto l, auto r) { return l < r ? T(3.0) : T(2.0); });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS, ewops_c<SCOLS, class c_ge>>(
      T(5.0), T(2.0), 3.0,
      [](auto l, auto r) { return l >= r ? T(3.0) : T(2.0); });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS, ewops_c<SCOLS, class c_le>>(
      T(5.0), T(2.0), 2.0,
      [](auto l, auto r) { return l <= r ? T(3.0) : T(2.0); });
}

int main() {
  static constexpr size_t TM = 8;
  static constexpr size_t TK = 16;

  static constexpr size_t MATRIX_M = TM * 2;
  static constexpr size_t MATRIX_N = 16 * 2;
  static constexpr size_t MATRIX_K = TK * 2;
  queue q;
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();
  for (int i = 0; i < combinations.size(); i++) {
    if (combinations[i].atype == matrix_type::bf16) {
      test_ewops_a<bfloat16, MATRIX_M, MATRIX_K, TM, TK>();
      if (combinations[i].nsize == 0 || combinations[i].nsize == 16)
        test_ewops_c<float, MATRIX_M, MATRIX_N, TM, 16>();
      else
        test_ewops_c<float, MATRIX_M, MATRIX_N, TM, 8>();
      return 0;
    }
  }
  return 0;
}
