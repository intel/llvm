#define TM 8
#define TK 32

class imatrix;

template <typename T, size_t M, size_t N>
void assert_ops_ref(host_accessor<T, 2, access::mode::read> C, const int ref) {
  for (size_t i = 0; i < M; i++)
    for (size_t j = 0; j < N; j++) {
      auto diff = C[i][j] - ref;
      assert(std::fabs(static_cast<int>(diff)) <=
             std::numeric_limits<int>::epsilon());
    }
}

template <typename T, size_t M, size_t N, typename OP>
void matrix_verify_op(big_matrix<T, M, N> &A, const int ref, OP op) {
  buffer<T, 2> bufB(A.get_data(), range<2>(M, N));

  queue q;
  size_t wg_size = get_wg_size<imatrix>(q);
  nd_range<2> r({M / TM, N / TN * wg_size}, {1, 1 * wg_size});

  q.submit([&](handler &cgh) {
     auto accA = bufB.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for<class imatrix>(
         r, [accA](nd_item<2> spmd_item)
#ifdef SG_SZ
                [[intel::reqd_sub_group_size(SG_SZ)]]
#endif
         {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T, use::b, TK, TN, layout::ext_intel_packed>
               sub_b;

           joint_matrix_fill(sg, sub_b, 5);

           joint_matrix_apply(sg, sub_b, op);
           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_b,
               accA.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N * 4 + sg_starty / wg_size * TN * 4,
               N * 4);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, M, N>(bufB.get_host_access(read_only), ref);
}

int main() {
  static constexpr size_t MATRIX_M = TM * 2;
  static constexpr size_t MATRIX_N = TN * 2;
  int8_t B[MATRIX_M][MATRIX_N];
  big_matrix<int8_t, MATRIX_M, MATRIX_N> MB((int8_t *)&B);

  matrix_verify_add<int8_t, MATRIX_M, MATRIX_N>(MB, 7,
                                                [](int8_t &x) { x = x + 2; });
  matrix_verify_sub<int8_t, MATRIX_M, MATRIX_N>(MB, 3,
                                                [](int8_t &x) { x = x - 2; });
  matrix_verify_mul<int8_t, MATRIX_M, MATRIX_N>(MB, 15,
                                                [](int8_t &x) { x = x * 3; });
  matrix_verify_div<int8_t, MATRIX_M, MATRIX_N>(MB, 2,
                                                [](int8_t &x) { x = x / 2; });
  matrix_verify_logic<int8_t, MATRIX_M, MATRIX_N>(MB, 7, [](T &x) {
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

  return 0;
}
