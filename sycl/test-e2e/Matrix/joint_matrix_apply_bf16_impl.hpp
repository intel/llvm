//===---joint_matrix_apply_bf16_impl.hpp - DPC++ joint_matrix--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

template <typename T, size_t TM, size_t TK> class add;
template <typename T, size_t TM, size_t TK> class add_func;

template <typename T> struct apply_add {
  void operator()(T &x) const { x = x + T(2); }
};

template <typename T, typename TResult, size_t Rows, size_t Cols,
          size_t TileRows, size_t TileCols, typename kernel_name, typename F>
void matrix_verify_add(big_matrix<T, Rows, Cols> &A, const TResult ref,
                       F &&lambda) {
  buffer<T, 2> bufA(A.get_data(), range<2>(Rows, Cols));

  queue q;
  size_t sg_size = get_sg_size<kernel_name>(q);
  nd_range<2> r({Rows / TileRows, Cols / TileCols * sg_size}, {1, 1 * sg_size});

  q.submit([&](handler &cgh) {
     accessor accA{bufA, cgh};

     cgh.parallel_for<kernel_name>(
         r, [=](nd_item<2> spmd_item)
#ifdef SG_SZ
                [[intel::reqd_sub_group_size(SG_SZ)]]
#endif
         {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T, use::a, TileRows, TileCols,
                        layout::row_major>
               sub_a;

           joint_matrix_fill(sg, sub_a, T(5.0));

           joint_matrix_apply(sg, sub_a, lambda);

           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_a,
               accA.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TileRows) * Cols +
                   sg_starty / sg_size * TileCols,
               Cols);
         }); // parallel for
   }).wait();
  // Check if the results are correct
  {
    host_accessor Acc{bufA};
    assert(std::all_of(Acc.begin(), Acc.end(), [=](auto Elem) {
      return (std::fabs(static_cast<TResult>(make_fp32(Elem) - ref)) <
              std::numeric_limits<TResult>::epsilon());
    }));
  }
}

template <typename T, typename TResult, size_t TM, size_t TK> void test() {
  std::cout << "Testing: " << TM << " x " << TK << " [TM x TK]" << std::endl;

  static constexpr size_t Rows = TM * 2;
  static constexpr size_t Cols = TK * 2;
  T A[Rows][Cols];

  big_matrix<T, Rows, Cols> MA((T *)&A);

  matrix_verify_add<T, TResult, Rows, Cols, TM, TK, add<T, TM, TK>>(
      MA, 7.0, [=](T &x) { x = x + T(2); });
  matrix_verify_add<T, TResult, Rows, Cols, TM, TK, add_func<T, TM, TK>>(
      MA, 7.0, apply_add<T>());
}

int main() {
  queue q;
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();

  for (unsigned int i = 0; i < combinations.size(); i++) {
    if (combinations[i].nsize == 0) { // Intel AMX
      test<bfloat16, float, /*TM*/ 16, /*TK*/ 32>();
      break;
    }

    if (combinations[i].nsize == 16) { // architecture::intel_gpu_pvc
      test<bfloat16, float, /*TM*/ 8, /*TK*/ 16>();
      // This combination is not currently supported for sub group size = 32 in
      // IGC
#if (!defined(SG_SZ) || SG_SZ != 32)
      test<bfloat16, float, /*TM*/ 32, /*TK*/ 16>();
      test<bfloat16, float, /*TM*/ 1, /*TK*/ 16>();
      test<bfloat16, float, /*TM*/ 16, /*TK*/ 16>();
#endif
      break;
    }

    if (combinations[i].nsize == 8) { // architecture::intel_gpu_dg2*
      test<bfloat16, float, /*TM*/ 8, /*TK*/ 16>();
      break;
    }
  }
  return 0;
}
