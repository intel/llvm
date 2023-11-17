#define TM 8
#define TK 16

template <typename T> struct apply_add {
  void operator()(T &x) const { x = x + bfloat16(2); }
};

template <typename T, size_t M, size_t K, class kernel_name, typename F>
void matrix_verify_add(big_matrix<T, M, K> &A, const float ref, F &&lambda) {
  buffer<bfloat16, 2> bufA(A.get_data(), range<2>(M, K));

  queue q;
  size_t wg_size = get_wg_size<kernel_name>(q);
  nd_range<2> r({M / TM, K / TK * wg_size}, {1, 1 * wg_size});

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
           joint_matrix<sub_group, T, use::a, TM, TK, layout::row_major> sub_a;
           joint_matrix_fill(sg, sub_a, bfloat16(5.0));
           joint_matrix_apply(sg, sub_a, lambda);

           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_a,
               accA.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * K + sg_starty / wg_size * TK,
               K);
         }); // parallel for
   }).wait();
  // Check if the results are correct
  {
    host_accessor Acc{bufA};
    assert(std::all_of(Acc.begin(), Acc.end(), [=](auto Elem) {
      return (std::fabs(static_cast<float>(make_fp32(Elem) - ref)) <
              std::numeric_limits<float>::epsilon());
    }));
  }
}

int main() {
  static constexpr size_t MATRIX_M = TM * 2;
  static constexpr size_t MATRIX_K = TK * 2;
  bfloat16 A[MATRIX_M][MATRIX_K];
  big_matrix<bfloat16, MATRIX_M, MATRIX_K> MA((bfloat16 *)&A);

  matrix_verify_add<bfloat16, MATRIX_M, MATRIX_K, class add1>(
      MA, 7.0, [=](bfloat16 &x) { x = x + bfloat16(2); });
  matrix_verify_add<bfloat16, MATRIX_M, MATRIX_K, class add2>(
      MA, 7.0, apply_add<bfloat16>());
  std::cout << "Passed\n";
  return 0;
}
