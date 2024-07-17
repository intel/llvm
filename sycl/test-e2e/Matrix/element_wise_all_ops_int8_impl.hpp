//===---element_wise_all_ops_int8_impl.hpp - DPC++ joint_matrix------------===//
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

template <typename T, size_t Rows, size_t Cols, typename R>
void assert_ops_ref(host_accessor<T, 2, access::mode::read> C, const R ref) {
  for (size_t i = 0; i < Rows; i++)
    for (size_t j = 0; j < Cols; j++) {
      auto diff = C[i][j] - ref;
      assert(std::fabs(static_cast<R>(diff)) <=
             std::numeric_limits<R>::epsilon());
    }
}

template <typename T, size_t Rows, size_t Cols, size_t TileRows,
          size_t TileCols, class kernel_name, typename R, typename OP>
void matrix_verify_op(big_matrix<T, Rows, Cols> &A, const R ref, OP op) {
  buffer<T, 2> bufA(A.get_data(), range<2>(Rows, Cols));

  queue q;
  size_t sg_size = get_sg_size<kernel_name>(q);
  nd_range<2> r({Rows / TileRows, Cols / TileCols * sg_size}, {1, 1 * sg_size});

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
           joint_matrix<sub_group, T, use::a, TileRows, TileCols,
                        layout::row_major>
               sub_a;

           joint_matrix_fill(sg, sub_a, 5);

           joint_matrix_apply(sg, sub_a, op);
           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_a,
               accA.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TileRows) * Cols +
                   sg_starty / sg_size * TileCols,
               Cols);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, Rows, Cols, R>(bufA.get_host_access(read_only), ref);
}

template <typename Ta, typename TResult, size_t TM, size_t TK> void test() {
  static constexpr size_t Rows = TM * 2;
  static constexpr size_t Cols = TK * 2;
  Ta A[Rows][Cols];

  big_matrix<Ta, Rows, Cols> MA((Ta *)&A);

  matrix_verify_op<Ta, Rows, Cols, TM, TK, add<TM, TK>, TResult>(
      MA, 7, [=](auto &x) { x = x + 2; });
  matrix_verify_op<Ta, Rows, Cols, TM, TK, sub<TM, TK>, TResult>(
      MA, 3, [=](auto &x) { x = x - 2; });
  matrix_verify_op<Ta, Rows, Cols, TM, TK, mul<TM, TK>, TResult>(
      MA, 10, [=](auto &x) { x = x * 2; });
  matrix_verify_op<Ta, Rows, Cols, TM, TK, divide<TM, TK>, TResult>(
      MA, 2, [=](auto &x) { x = x / 2; }); // truncation is expected
  matrix_verify_op<Ta, Rows, Cols, TM, TK, logic<TM, TK>, TResult>(
      MA, 7, [=](auto &x) {
        if (x) {
          if (x > 2 || x >= 2 || x < 2 || x <= 2) {
            Ta val = (x != 2) ? x : 2;
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
      test<int8_t, int, /*TM*/ 16, /*TK*/ 64>();
      break;
    }

    if (combinations[i].nsize == 16) { // architecture::intel_gpu_pvc
      test<int8_t, int, /*TM*/ 8, /*TK*/ 32>();
      break;
    }

    if (combinations[i].nsize == 8) { // architecture::intel_gpu_dg2*
      test<int8_t, int, /*TM*/ 8, /*TK*/ 32>();
      break;
    }
  }

  return 0;
}
