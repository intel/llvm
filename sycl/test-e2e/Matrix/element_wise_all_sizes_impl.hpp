//===---element_wise_all_ops_all_sizes_impl.hpp - DPC++ joint_matrix-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

static constexpr size_t M_MULTIPLIER = 16;
template <typename T, size_t TileM, size_t TileN, size_t TileK> class add;

template <typename T, size_t M, size_t N>
void assert_ops_ref(host_accessor<T, 2, access::mode::read_write> C,
                    const T ref) {
  for (size_t i = 0; i < M; i++)
    for (size_t j = 0; j < N; j++) {
      if (std::is_same_v<T, bfloat16>) {
        auto diff = make_fp32(C[i][j]) - make_fp32(ref);
        assert(std::fabs(static_cast<float>(diff)) <
               std::numeric_limits<float>::epsilon());
      } else if (std::is_same_v<T, int8_t>) {
        assert(C[i][j] == ref);
      }
    }
}

template <typename T, typename T1, size_t TM, size_t TN, size_t TK,
          typename kernel_name>
void matrix_verify_add(const T1 val1, const T1 val2, const T1 result) {
  static constexpr size_t M = TM * M_MULTIPLIER;
  static constexpr size_t K = 128;
  T MatA[M][K];

  size_t NDRangeM = M / TM;
  size_t NDRangeK = K / TK;
  queue q;
  size_t sg_size = get_sg_size<kernel_name>(q);
  nd_range<2> r({NDRangeM, NDRangeK * sg_size}, {1, 1 * sg_size});
  big_matrix<T, M, K> A((T *)&MatA);

  buffer<T, 2> bufA(A.get_data(), range<2>(M, K));

  q.submit([&](handler &cgh) {
     sycl::accessor accA{bufA, cgh, sycl::read_write};

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
           joint_matrix<sub_group, T, use::a, TM, TK, layout::row_major> sub_a;

           joint_matrix_fill(sg, sub_a, val1);

           joint_matrix_apply(sg, sub_a, [=](T &x) { x += val2; });

           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_a,
               accA.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * K + sg_starty / sg_size * TK,
               K);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, M, K>(bufA.get_host_access(), result);
}

template <typename Ta, size_t tM, size_t tN, size_t tK, typename kernel_name>
void add_ref() {
  if constexpr (std::is_same_v<Ta, bfloat16>) {
    // Tests whether 5 + 2 = 7 operation is successful.
    matrix_verify_add<bfloat16, bfloat16, tM, tN, tK, kernel_name>(
        bfloat16(5.0), bfloat16(2.0), bfloat16(7.0));
  }
  if constexpr (std::is_same_v<Ta, int8_t>) {
    matrix_verify_add<int8_t, int, tM, tN, tK, kernel_name>(
        5 /*val1*/, 2 /*val2*/, 7 /*result*/);
  }
}

template <size_t TN> void test() {
  add_ref<bfloat16, 1, TN, 16, add<bfloat16, 1, TN, 16>>();
  add_ref<bfloat16, 2, TN, 16, add<bfloat16, 2, TN, 16>>();
  add_ref<bfloat16, 3, TN, 16, add<bfloat16, 3, TN, 16>>();
  add_ref<bfloat16, 4, TN, 16, add<bfloat16, 4, TN, 16>>();
  add_ref<bfloat16, 5, TN, 16, add<bfloat16, 5, TN, 16>>();
  add_ref<bfloat16, 6, TN, 16, add<bfloat16, 6, TN, 16>>();
  add_ref<bfloat16, 7, TN, 16, add<bfloat16, 7, TN, 16>>();

  add_ref<int8_t, 1, TN, 32, add<int8_t, 1, TN, 32>>();
  add_ref<int8_t, 2, TN, 32, add<int8_t, 2, TN, 32>>();
  add_ref<int8_t, 3, TN, 32, add<int8_t, 3, TN, 32>>();
  add_ref<int8_t, 4, TN, 32, add<int8_t, 4, TN, 32>>();
  add_ref<int8_t, 5, TN, 32, add<int8_t, 5, TN, 32>>();
  add_ref<int8_t, 6, TN, 32, add<int8_t, 6, TN, 32>>();
  add_ref<int8_t, 7, TN, 32, add<int8_t, 7, TN, 32>>();
}

int main() {
  queue q;
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();

  for (unsigned int i = 0; i < combinations.size(); i++) {
    if (combinations[i].nsize == 0) { // Intel AMX
      test<16>();
      break;
    }

    if (combinations[i].nsize == 16) { // architecture::intel_gpu_pvc
      test<16>();
      break;
    }

    if (combinations[i].nsize == 8) { // architecture::intel_gpu_dg2*
      test<8>();
      break;
    }
  }

  std::cout << "Passed\n";
}
