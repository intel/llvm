//===---element_wise_all_ops_int8_impl.hpp - DPC++ joint_matrix------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define TM 8
#define TK 32

template <typename T, size_t M, size_t N, typename R>
void assert_ops_ref(host_accessor<T, 2, access::mode::read> C, const R ref) {
  for (size_t i = 0; i < M; i++)
    for (size_t j = 0; j < N; j++) {
      auto diff = C[i][j] - ref;
      assert(std::fabs(static_cast<R>(diff)) <=
             std::numeric_limits<R>::epsilon());
    }
}

template <typename T, size_t M, size_t N, size_t TileM, size_t TileN,
          size_t TileK, class kernel_name, typename R, typename OP>
void matrix_verify_op(big_matrix<T, M, N> &A, const R ref, OP op) {
  buffer<int8_t, 2> bufA(A.get_data(), range<2>(M, N));

  queue q;
  size_t sg_size = get_sg_size<kernel_name>(q);
  nd_range<2> r({M / TileM, N / TileN * sg_size}, {1, 1 * sg_size});

  q.submit([&](handler &cgh) {
     auto accA = bufA.get_access<access::mode::read_write>(cgh);

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
           joint_matrix<sub_group, T, use::a, TileM, TileK, layout::row_major>
               sub_a;

           joint_matrix_fill(sg, sub_a, 5);

           joint_matrix_apply(sg, sub_a, op);
           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_a,
               accA.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TileM) * N + sg_starty / sg_size * TileN,
               N);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, M, N, R>(bufA.get_host_access(read_only), ref);
}

static constexpr size_t MATRIX_M = TM * 2;
static constexpr size_t MATRIX_N = TN * 2;
int8_t A[MATRIX_M][MATRIX_N];

int main() {

  big_matrix<int8_t, MATRIX_M, MATRIX_N> MA((int8_t *)&A);

  matrix_verify_op<int8_t, MATRIX_M, MATRIX_N, TM, TN, TK, class add, int>(
      MA, 7, [=](auto &x) { x = x + 2; });
  matrix_verify_op<int8_t, MATRIX_M, MATRIX_N, TM, TN, TK, class sub, int>(
      MA, 3, [=](auto &x) { x = x - 2; });
  matrix_verify_op<int8_t, MATRIX_M, MATRIX_N, TM, TN, TK, class mul, int>(
      MA, 10, [=](auto &x) { x = x * 2; });
  matrix_verify_op<int8_t, MATRIX_M, MATRIX_N, TM, TN, TK, class div, int>(
      MA, 2, [=](auto &x) { x = x / 2; }); // truncation is expected
  matrix_verify_op<int8_t, MATRIX_M, MATRIX_N, TM, TN, TK, class logic, int>(
      MA, 7, [=](auto &x) {
        if (x) {
          if (x > 2 || x >= 2 || x < 2 || x <= 2) {
            int8_t val = (x != 2) ? x : 2;
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

  return 0;
}
