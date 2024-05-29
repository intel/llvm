//===---element_wise_all_ops_int8_packed_impl.hpp - DPC++ joint_matrix-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

template <size_t TileRows, size_t TileCols> class add;
template <size_t TileRows, size_t TileCols> class sub;
template <size_t TileRows, size_t TileCols> class mul;
template <size_t TileRows, size_t TileCols> class divide;
template <size_t TileRows, size_t TileCols> class logic;

template <typename T, size_t Rows, size_t Cols, typename TResult>
void assert_ops_ref(host_accessor<T, 2, access::mode::read> C,
                    const TResult ref) {
  for (size_t i = 0; i < Rows; i++)
    for (size_t j = 0; j < Cols; j++) {
      TResult diff = C[i][j] - ref;
      assert(std::fabs(static_cast<TResult>(diff)) <=
             std::numeric_limits<TResult>::epsilon());
    }
}

template <typename T, size_t Rows, size_t Cols, size_t TileRows,
          size_t TileCols, size_t VNNI, class kernel_name, typename TResult,
          typename OP>
void matrix_verify_op(big_matrix<T, Rows, Cols> &B, const TResult ref, OP op) {
  buffer<T, 2> bufB(B.get_data(), range<2>(Rows, Cols));

  queue q;
  size_t sg_size = get_sg_size<kernel_name>(q);
  nd_range<2> r({Rows / TileRows, Cols / TileCols * sg_size}, {1, 1 * sg_size});

  q.submit([&](handler &cgh) {
     sycl::accessor accB{bufB, cgh, sycl::read_write};

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
           joint_matrix<sub_group, T, use::b, TileRows, TileCols,
                        layout::ext_intel_packed>
               sub_b;

           joint_matrix_fill(sg, sub_b, 5);

           joint_matrix_apply(sg, sub_b, op);
           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_b,
               accB.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TileRows / VNNI) * Cols * VNNI +
                   sg_starty / sg_size * TileCols * VNNI,
               Cols * VNNI);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, Rows, Cols, TResult>(bufB.get_host_access(read_only), ref);
}

template <typename T, typename TResult, size_t TK, size_t TN, size_t VNNI>
void test() {
  static constexpr size_t Rows = TK * 2;
  static constexpr size_t Cols = TN * 2;
  T B[Rows][Cols];

  big_matrix<T, Rows, Cols> MB((T *)&B);

  matrix_verify_op<T, Rows, Cols, TK, TN, VNNI, add<TK, TN>, TResult>(
      MB, 7, [=](auto &x) { x = x + 2; });
  matrix_verify_op<T, Rows, Cols, TK, TN, VNNI, sub<TK, TN>, TResult>(
      MB, 3, [=](auto &x) { x = x - 2; });
  matrix_verify_op<T, Rows, Cols, TK, TN, VNNI, mul<TK, TN>, TResult>(
      MB, 10, [=](auto &x) { x = x * 2; });
  matrix_verify_op<T, Rows, Cols, TK, TN, VNNI, divide<TK, TN>, TResult>(
      MB, 2, [=](auto &x) { x = x / 2; }); // truncation is expected
  matrix_verify_op<T, Rows, Cols, TK, TN, VNNI, logic<TK, TN>, TResult>(
      MB, 7, [=](auto &x) {
        if (x) {
          if (x > 2 || x >= 2 || x < 2 || x <= 2) {
            T val = (x != 2) ? x : 2;
            val--;
            val++;
            if (x == 2) {
              val -= 2;
              val *= 3;
              val /= 2;
            } else {
              val += 2;
            }
            x = val;
          }
        }
      });
}

int main() {
  queue q;
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();

  for (unsigned int i = 0; i < combinations.size(); i++) {
    if (combinations[i].nsize == 0) { // Intel AMX
      test<int8_t, int, /*TK*/ 64, /*TN*/ 16, /*VNNI*/ 4>();
      break;
    }

    if (combinations[i].nsize == 16) { // architecture::intel_gpu_pvc
      test<int8_t, int, /*TK*/ 32, /*TN*/ 16, /*VNNI*/ 4>();
      break;
    }

    if (combinations[i].nsize == 8) { // architecture::intel_gpu_dg2*
      test<int8_t, int, /*TK*/ 32, /*TN*/ 8, /*VNNI*/ 4>();
      break;
    }
  }

  return 0;
}
