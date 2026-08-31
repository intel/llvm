#include <iostream>
//==----------- element_wise_all_ops_impl.hpp  - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <sycl/usm.hpp>

template <typename T, size_t NUM_ROWS, size_t NUM_COLS>
void assert_ops_ref(host_accessor<T, 2, access_mode::read> mat,
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

// Overload for results held in a USM allocation instead of a buffer.
template <typename T, size_t NUM_ROWS, size_t NUM_COLS>
void assert_ops_ref(const T *mat, const float ref) {
  for (size_t i = 0; i < NUM_ROWS; i++)
    for (size_t j = 0; j < NUM_COLS; j++) {
      float diff;
      if constexpr (std::is_same_v<T, bfloat16>)
        diff = make_fp32(mat[i * NUM_COLS + j]) - ref;
      else
        diff = mat[i * NUM_COLS + j] - ref;
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
             [[sycl::reqd_sub_group_size(SG_SZ)]]
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
          size_t SUB_COLS, use Use, layout Layout, size_t VF, class kernel_name,
          typename OP, std::enable_if_t<is_fp8_type_v<T>, bool> = true>
void verify_op_ab(const sycl::half l, const sycl::half r, const float ref,
                  OP op) {
  queue q;
  size_t sg_size = get_sg_size<kernel_name>(q);

  static constexpr size_t Rows = NUM_ROWS / VF;
  static constexpr size_t Cols = NUM_COLS * VF;

  // The 8-bit float types only convert to half on the device, so the results
  // are converted by matrix_copy's kernel rather than on the host. matrix_copy
  // submits that kernel itself, so it must be called outside of any command
  // group and both matrices have to live in USM.
  T *mat = sycl::malloc_shared<T>(Rows * Cols, q);
  sycl::half *matH = sycl::malloc_shared<sycl::half>(Rows * Cols, q);

  q.submit([&](handler &cgh) {
     cgh.parallel_for<kernel_name>(
         nd_range<2>({NUM_ROWS / SUB_ROWS, NUM_COLS / SUB_COLS * sg_size},
                     {1, 1 * sg_size}),
         [=](nd_item<2> spmd_item)
#ifdef SG_SZ
             [[sycl::reqd_sub_group_size(SG_SZ)]]
#endif
         {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T, Use, SUB_ROWS, SUB_COLS, Layout> sub_mat;
           joint_matrix_fill(sg, sub_mat, T(l));
           joint_matrix_apply(sg, sub_mat, [=](T &x) { x = op((half)x, r); });
           auto pMat =
               address_space_cast<sycl::access::address_space::global_space,
                                  access::decorated::no>(mat);
           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_mat,
               pMat + (sg_startx * SUB_ROWS / VF) * Cols +
                   sg_starty / sg_size * SUB_COLS * VF,
               Cols);
         }); // parallel for
   }).wait();

  matrix_copy<T, sycl::half>(q, Rows, Cols, mat, matH);

  assert_ops_ref<sycl::half, Rows, Cols>(matH, ref);

  sycl::free(mat, q);
  sycl::free(matH, q);
}

template <typename T, size_t NUM_ROWS, size_t NUM_COLS, size_t SUB_ROWS,
          size_t SUB_COLS, use Use, layout Layout, size_t VF, class kernel_name,
          size_t numElems = 2, typename OP,
          std::enable_if_t<is_4bit_type_v<T, numElems>, bool> = true>
void verify_op_ab(const sycl::half l, const sycl::half r, const float ref,
                  OP op) {
  queue q;
  size_t sg_size = get_sg_size<kernel_name>(q);

  // Rows/Cols count packed elements: each T holds numElems values, so a row of
  // NUM_COLS values takes NUM_COLS / numElems of them, and the packed layout
  // folds VF rows into one row of VF times the width. Cols is therefore the row
  // stride in packed fp4_e2m1_x<numElems> storage elements.
  // NOTE: the packed VNNI layout for 4-bit types is unverified, GSD-9057.
  static constexpr size_t Rows = NUM_ROWS / VF;
  static constexpr size_t Cols = NUM_COLS / numElems * VF;

  // As in the fp8 overload above, the conversion to half happens in
  // matrix_copy's own kernel, so both matrices live in USM.
  T *mat = sycl::malloc_shared<T>(Rows * Cols, q);
  sycl::half *matH = sycl::malloc_shared<sycl::half>(Rows * Cols * numElems, q);

  q.submit([&](handler &cgh) {
     cgh.parallel_for<kernel_name>(
         nd_range<2>({NUM_ROWS / SUB_ROWS, NUM_COLS / SUB_COLS * sg_size},
                     {1, 1 * sg_size}),
         [=](nd_item<2> spmd_item)
#ifdef SG_SZ
             [[sycl::reqd_sub_group_size(SG_SZ)]]
#endif
         {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T, Use, SUB_ROWS, SUB_COLS, Layout> sub_mat;
           marray<sycl::half, numElems> fillVal(l);
           joint_matrix_fill(sg, sub_mat, T(fillVal));
           joint_matrix_apply(sg, sub_mat, [=](T &x) {
             marray<sycl::half, numElems> mval =
                 (marray<sycl::half, numElems>)x;
             for (unsigned int p = 0; p < numElems; p++)
               mval[p] = op(mval[p], r);
             // The 4-bit types construct from an marray explicitly only.
             x = T(mval);
           });
           auto pMat =
               address_space_cast<sycl::access::address_space::global_space,
                                  access::decorated::no>(mat);
           // The row offset divides by VF, not by numElems: VF folds rows into
           // the packed layout, while numElems is already accounted for by Cols
           // being a count of packed elements.
           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_mat,
               pMat + (sg_startx * SUB_ROWS / VF) * Cols +
                   sg_starty / sg_size * SUB_COLS / numElems * VF,
               Cols);
         }); // parallel for
   }).wait();

  // matrix_copy expands each packed element into numElems halves, so it takes
  // packed columns and writes numElems times as many values.
  matrix_copy<T, sycl::half, numElems>(q, Rows, Cols, mat, matH);

  assert_ops_ref<sycl::half, Rows, Cols * numElems>(matH, ref);

  sycl::free(mat, q);
  sycl::free(matH, q);
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
             [[sycl::reqd_sub_group_size(SG_SZ)]]
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
          size_t VF, typename Tv = T>
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
      Tv(5.0), Tv(2.0), 7.0, [](auto l, auto r) { return l + r; });
  verify_op_ab<T, NROWS, NCOLS, SROWS, SCOLS, Use, Layout, VF,
               ewops_ab<T, SROWS, SCOLS, Use, class ab_sub>>(
      Tv(5.0), Tv(2.0), 3.0, [](auto l, auto r) { return l - r; });
  verify_op_ab<T, NROWS, NCOLS, SROWS, SCOLS, Use, Layout, VF,
               ewops_ab<T, SROWS, SCOLS, Use, class ab_mul>>(
      Tv(5.0), Tv(2.0), 10.0, [](auto l, auto r) { return l * r; });
  verify_op_ab<T, NROWS, NCOLS, SROWS, SCOLS, Use, Layout, VF,
               ewops_ab<T, SROWS, SCOLS, Use, class ab_div>>(
      Tv(5.0), Tv(2.0), 2.5, [](auto l, auto r) { return l / r; });
  verify_op_ab<T, NROWS, NCOLS, SROWS, SCOLS, Use, Layout, VF,
               ewops_ab<T, SROWS, SCOLS, Use, class ab_logical>>(
      Tv(5.0), Tv(5.0), 5.0,
      [](auto l, auto r) { return l == r ? l : Tv(1.0); });
  verify_op_ab<T, NROWS, NCOLS, SROWS, SCOLS, Use, Layout, VF,
               ewops_ab<T, SROWS, SCOLS, Use, class ab_eq>>(
      Tv(5.0), Tv(4.0), 4.0, [](auto l, auto r) { return l == r ? l : r; });
  verify_op_ab<T, NROWS, NCOLS, SROWS, SCOLS, Use, Layout, VF,
               ewops_ab<T, SROWS, SCOLS, Use, class ab_ne>>(
      Tv(5.0), Tv(5.0), 1.0,
      [](auto l, auto r) { return l != r ? l : Tv(1.0); });
  verify_op_ab<T, NROWS, NCOLS, SROWS, SCOLS, Use, Layout, VF,
               ewops_ab<T, SROWS, SCOLS, Use, class ab_gt>>(
      Tv(5.0), Tv(2.0), 3.0,
      [](auto l, auto r) { return l > r ? Tv(3.0) : Tv(2.0); });
  verify_op_ab<T, NROWS, NCOLS, SROWS, SCOLS, Use, Layout, VF,
               ewops_ab<T, SROWS, SCOLS, Use, class ab_lt>>(
      Tv(5.0), Tv(2.0), 2.0,
      [](auto l, auto r) { return l < r ? Tv(3.0) : Tv(2.0); });
  verify_op_ab<T, NROWS, NCOLS, SROWS, SCOLS, Use, Layout, VF,
               ewops_ab<T, SROWS, SCOLS, Use, class ab_ge>>(
      Tv(5.0), Tv(2.0), 3.0,
      [](auto l, auto r) { return l >= r ? Tv(3.0) : Tv(2.0); });
  verify_op_ab<T, NROWS, NCOLS, SROWS, SCOLS, Use, Layout, VF,
               ewops_ab<T, SROWS, SCOLS, Use, class ab_le>>(
      Tv(5.0), Tv(2.0), 2.0,
      [](auto l, auto r) { return l <= r ? Tv(3.0) : Tv(2.0); });
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
        test_ewops_ab<bfloat16, 1, 32, use::a, layout::row_major, 1>();
        test_ewops_ab<bfloat16, 32, 16, use::a, layout::row_major, 1>();
        test_ewops_ab<bfloat16, 32, 32, use::a, layout::row_major, 1>();
        test_ewops_ab<bfloat16, 16, 64, use::b, layout::ext_intel_packed, 2>();
        test_ewops_ab<bfloat16, 32, 64, use::b, layout::ext_intel_packed, 2>();
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
      test_ewops_ab<bfloat16, 32, 16, use::a, layout::row_major, 1>();
      test_ewops_ab<bfloat16, 16, 32, use::b, layout::ext_intel_packed, 2>();
      test_ewops_c<float, 32, 32>();
      break;
    }
  }
  // fp4_e2m1_x packs 1 or 2 elements per byte, so the 4-bit tests run at a
  // packing factor of 2.
  constexpr unsigned int numElems = 2;
  if (is_type_supported_by_device(q, matrix_type::fp4_e2m1)) {
    // The advertised fp4_e2m1 combination is {msize=8, nsize=16, ksize=32} with
    // no max_* sizes, so use::a and use::b extents have to match it exactly.
    //
    // VF counts logical elements, i.e. 32 bits / element bits: 2 for bfloat16,
    // 4 for the 8-bit float types, 8 here. Eight 4-bit values are one 32-bit
    // dword, so a dword holds VF consecutive k of a single B column. numElems
    // is a separate axis: it only converts the logical column extent into a
    // count of packed storage elements. This layout cannot be verified until
    // IGC implements a 4-bit DPAS -- it currently rejects the i4 cooperative
    // matrix component outright. Tracked by GSD-9057.
    //
    // If that instruction turns out to want the
    // alternative format, where data is packed along the row first and whole
    // bytes are folded afterwards, then VF becomes 4 and the host-side
    // pack/fold order in joint_matrix_float4_impl.hpp has to change with it;
    // the two are not independent.
    test_ewops_ab<syclex::fp4_e2m1_x<numElems>, 8, 32, use::a,
                  layout::row_major, 1, sycl::half>();
    test_ewops_ab<syclex::fp4_e2m1_x<numElems>, 32, 16, use::b,
                  layout::ext_intel_packed, 8, sycl::half>();
  }

  if (is_type_supported_by_device(q, matrix_type::fp8_e5m2)) {
    test_ewops_ab<syclex::fp8_e5m2, 8, 32, use::a, layout::row_major, 1,
                  sycl::half>();
    test_ewops_ab<syclex::fp8_e4m3, 8, 32, use::a, layout::row_major, 1,
                  sycl::half>();
    test_ewops_ab<syclex::fp8_e5m2, 32, 16, use::b, layout::ext_intel_packed, 4,
                  sycl::half>();
    test_ewops_ab<syclex::fp8_e4m3, 32, 16, use::b, layout::ext_intel_packed, 4,
                  sycl::half>();
  }

  return 0;
}
