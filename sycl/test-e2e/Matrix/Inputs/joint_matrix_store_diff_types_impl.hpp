//===---joint_matrix_store_diff_types_impl.hpp - DPC++ joint_matrix--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/usm.hpp>

template <typename T, size_t NUM_ROWS, size_t NUM_COLS>
void assert_ref(T *mat, const float ref) {
  for (size_t i = 0; i < NUM_ROWS; i++)
    for (size_t j = 0; j < NUM_COLS; j++) {
      auto diff = mat[i * NUM_COLS + j] - ref;
      assert(std::fabs(static_cast<float>(diff)) <
             std::numeric_limits<float>::epsilon());
    }
}

template <typename T, unsigned int ROWS, unsigned int COLS> class st;

template <typename Tp, size_t SUB_ROWS, size_t SUB_COLS>
void store(const float ref) {

  queue q;
  static constexpr size_t NUM_ROWS = SUB_ROWS * 2;
  static constexpr size_t NUM_COLS = SUB_COLS * 2;
  std::cout << SUB_ROWS << "x" << SUB_COLS << "\n";

  Tp *A = sycl::malloc_shared<Tp>(NUM_ROWS * NUM_COLS, q);
  size_t sg_size = get_sg_size<st<Tp, SUB_ROWS, SUB_COLS>>(q);
  q.submit([&](handler &cgh) {
     cgh.parallel_for<st<Tp, SUB_ROWS, SUB_COLS>>(
         nd_range<2>({NUM_ROWS / SUB_ROWS, NUM_COLS / SUB_COLS * sg_size},
                     {1, 1 * sg_size}),
         [=](nd_item<2> spmd_item)
#ifdef SG_SZ
             [[sycl::reqd_sub_group_size(SG_SZ)]]
#endif
         {
           auto pA =
               address_space_cast<sycl::access::address_space::global_space,
                                  access::decorated::yes>(A);

           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, float, use::accumulator, SUB_ROWS, SUB_COLS>
               sub_mat;
           joint_matrix_fill(sg, sub_mat, ref);
           joint_matrix_store(sg, sub_mat,
                              pA + (sg_startx * SUB_ROWS) * NUM_COLS +
                                  sg_starty / sg_size * SUB_COLS,
                              NUM_COLS, layout::row_major);
         }); // parallel for
   }).wait();
  assert_ref<Tp, NUM_ROWS, NUM_COLS>(A, ref);
}

template <typename T, size_t ROWS, size_t COLS, class name> class ewops_c {};

int main() {
  queue q;
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();

  for (auto &combination : combinations) {
    if (combination.nsize == 0 ||
        combination.nsize == 16) { // Intel AMX or architecture::intel_gpu_pvc

      store<half, 8, 16>(7.0);
      store<bfloat16, 8, 16>(7.0);
      break;
    }
    if (combination.nsize == 8) { // architecture::intel_gpu_dg2*
      store<half, 8, 8>(7.0);
      store<bfloat16, 8, 8>(7.0);
      break;
    }
  }
  return 0;
}
