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
          size_t SUB_COLS, use Use, layout Layout, size_t VF, class kernel_name,
          typename OP>
void verify_op_ab(const T l, const T r, const float ref, OP op) {
  T mat[NUM_ROWS / VF][NUM_COLS * VF];
  big_matrix<T, NUM_ROWS / VF, NUM_COLS * VF> big_mat((T *)&mat);

  buffer<T, 2> bufMat(big_mat.get_data(),
                      range<2>(NUM_ROWS / VF, NUM_COLS * VF));

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
           joint_matrix<sub_group, T, Use, SUB_ROWS, SUB_COLS, Layout> sub_mat;
           joint_matrix_fill(sg, sub_mat, l);
           joint_matrix_apply(sg, sub_mat, [=](T &x) { x = op(x, r); });
           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_mat,
               accessMat.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * SUB_ROWS / VF) * NUM_COLS * VF +
                   sg_starty / sg_size * SUB_COLS * VF,
               NUM_COLS * VF);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, NUM_ROWS / VF, NUM_COLS * VF>(
      bufMat.get_host_access(read_only), ref);
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

// Avoid same kernel name for different types
template <typename T, size_t SROWS, size_t SCOLS, use Use, class name>
class ewops_ab {};
template <typename T, size_t SROWS, size_t SCOLS, use Use, layout Layout,
          size_t VF>
void test_ewops_ab() {
  if constexpr (Use == use::a)
    std::cout << "Test A ";
  else
    std::cout << "Test B ";
  std::cout << SROWS << "x" << SCOLS << "\n";

  static constexpr size_t NROWS = SROWS * 2;
  static constexpr size_t NCOLS = SCOLS * 2;

  verify_op_ab<T, NROWS, NCOLS, SROWS, SCOLS, Use, Layout, VF,
               ewops_ab<T, SROWS, SCOLS, Use, class ab_add>>(
      T(5.0), T(2.0), 7.0, [](auto l, auto r) { return l + r; });
  verify_op_ab<T, NROWS, NCOLS, SROWS, SCOLS, Use, Layout, VF,
               ewops_ab<T, SROWS, SCOLS, Use, class ab_sub>>(
      T(5.0), T(2.0), 3.0, [](auto l, auto r) { return l - r; });
  verify_op_ab<T, NROWS, NCOLS, SROWS, SCOLS, Use, Layout, VF,
               ewops_ab<T, SROWS, SCOLS, Use, class ab_mul>>(
      T(5.0), T(2.0), 10.0, [](auto l, auto r) { return l * r; });
  verify_op_ab<T, NROWS, NCOLS, SROWS, SCOLS, Use, Layout, VF,
               ewops_ab<T, SROWS, SCOLS, Use, class ab_div>>(
      T(5.0), T(2.0), 2.5, [](auto l, auto r) { return l / r; });
  verify_op_ab<T, NROWS, NCOLS, SROWS, SCOLS, Use, Layout, VF,
               ewops_ab<T, SROWS, SCOLS, Use, class ab_logical>>(
      T(5.0), T(5.0), 5.0, [](auto l, auto r) { return l == r ? l : T(1.0); });
  verify_op_ab<T, NROWS, NCOLS, SROWS, SCOLS, Use, Layout, VF,
               ewops_ab<T, SROWS, SCOLS, Use, class ab_eq>>(
      T(5.0), T(4.0), 4.0, [](auto l, auto r) { return l == r ? l : r; });
  verify_op_ab<T, NROWS, NCOLS, SROWS, SCOLS, Use, Layout, VF,
               ewops_ab<T, SROWS, SCOLS, Use, class ab_ne>>(
      T(5.0), T(5.0), 1.0, [](auto l, auto r) { return l != r ? l : T(1.0); });
  verify_op_ab<T, NROWS, NCOLS, SROWS, SCOLS, Use, Layout, VF,
               ewops_ab<T, SROWS, SCOLS, Use, class ab_gt>>(
      T(5.0), T(2.0), 3.0,
      [](auto l, auto r) { return l > r ? T(3.0) : T(2.0); });
  verify_op_ab<T, NROWS, NCOLS, SROWS, SCOLS, Use, Layout, VF,
               ewops_ab<T, SROWS, SCOLS, Use, class ab_lt>>(
      T(5.0), T(2.0), 2.0,
      [](auto l, auto r) { return l < r ? T(3.0) : T(2.0); });
  verify_op_ab<T, NROWS, NCOLS, SROWS, SCOLS, Use, Layout, VF,
               ewops_ab<T, SROWS, SCOLS, Use, class ab_ge>>(
      T(5.0), T(2.0), 3.0,
      [](auto l, auto r) { return l >= r ? T(3.0) : T(2.0); });
  verify_op_ab<T, NROWS, NCOLS, SROWS, SCOLS, Use, Layout, VF,
               ewops_ab<T, SROWS, SCOLS, Use, class ab_le>>(
      T(5.0), T(2.0), 2.0,
      [](auto l, auto r) { return l <= r ? T(3.0) : T(2.0); });
}

// Avoid same kernel name for different types and numbers of columns
template <typename T, size_t ROWS, size_t COLS, class name> class ewops_c {};
template <typename T, size_t SROWS, size_t SCOLS> void test_ewops_c() {
  std::cout << "Test C " << SROWS << "x" << SCOLS << "\n";

  static constexpr size_t NROWS = SROWS * 2;
  static constexpr size_t NCOLS = SCOLS * 2;

  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS,
              ewops_c<T, SROWS, SCOLS, class c_add>>(
      T(5.0), T(2.0), 7.0, [](auto l, auto r) { return l + r; });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS,
              ewops_c<T, SROWS, SCOLS, class c_sub>>(
      T(5.0), T(2.0), 3.0, [](auto l, auto r) { return l - r; });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS,
              ewops_c<T, SROWS, SCOLS, class c_mul>>(
      T(5.0), T(2.0), 10.0, [](auto l, auto r) { return l * r; });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS,
              ewops_c<T, SROWS, SCOLS, class c_div>>(
      T(5.0), T(2.0), 2.5, [](auto l, auto r) { return l / r; });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS,
              ewops_c<T, SROWS, SCOLS, class c_logical>>(
      T(5.0), T(5.0), 5.0, [](auto l, auto r) { return l == r ? l : T(1.0); });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS,
              ewops_c<T, SROWS, SCOLS, class c_eq>>(
      T(5.0), T(4.0), 4.0, [](auto l, auto r) { return l == r ? l : r; });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS,
              ewops_c<T, SROWS, SCOLS, class c_ne>>(
      T(5.0), T(5.0), 1.0, [](auto l, auto r) { return l != r ? l : T(1.0); });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS,
              ewops_c<T, SROWS, SCOLS, class c_gt>>(
      T(5.0), T(2.0), 3.0,
      [](auto l, auto r) { return l > r ? T(3.0) : T(2.0); });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS,
              ewops_c<T, SROWS, SCOLS, class c_lt>>(
      T(5.0), T(2.0), 2.0,
      [](auto l, auto r) { return l < r ? T(3.0) : T(2.0); });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS,
              ewops_c<T, SROWS, SCOLS, class c_ge>>(
      T(5.0), T(2.0), 3.0,
      [](auto l, auto r) { return l >= r ? T(3.0) : T(2.0); });
  verify_op_c<T, NROWS, NCOLS, SROWS, SCOLS,
              ewops_c<T, SROWS, SCOLS, class c_le>>(
      T(5.0), T(2.0), 2.0,
      [](auto l, auto r) { return l <= r ? T(3.0) : T(2.0); });
}

int main() {
  queue q;
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();

  for (auto &combination : combinations) {
    if (combination.nsize == 0 ||
        combination.nsize == 16) { // Intel AMX or architecture::intel_gpu_pvc
      test_ewops_ab<bfloat16, 1, 16, use::a, layout::row_major, 1>();
      test_ewops_ab<bfloat16, 8, 16, use::a, layout::row_major, 1>();
      test_ewops_ab<bfloat16, 16, 16, use::b, layout::ext_intel_packed, 2>();
      test_ewops_c<float, 1, 16>();
      test_ewops_c<float, 8, 16>();

      if (combination.nsize == 16) { // architecture::intel_gpu_pvc
        test_ewops_ab<bfloat16, 16, 16, use::a, layout::row_major, 1>();
        test_ewops_c<float, 16, 16>();
// This combination is not currently supported for sub group size = 32 in IGC
#if (!defined(SG_SZ) || SG_SZ != 32)
        test_ewops_ab<bfloat16, 32, 16, use::a, layout::row_major, 1>();
        test_ewops_ab<bfloat16, 16, 64, use::b, layout::ext_intel_packed, 2>();
        test_ewops_c<float, 1, 64>();
        test_ewops_c<float, 32, 64>();
#endif
      }
      break;
    }

    if (combination.nsize == 8) { // architecture::intel_gpu_dg2*
      test_ewops_ab<bfloat16, 8, 16, use::a, layout::row_major, 1>();
      test_ewops_ab<bfloat16, 16, 8, use::b, layout::ext_intel_packed, 2>();
      test_ewops_c<float, 8, 8>();
      break;
    }
  }

  return 0;
}
