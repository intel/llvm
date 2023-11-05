using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

static constexpr size_t M_MULTIPLIER = 16;

template <typename T1, typename T2, size_t M, size_t N, size_t K,
          int vnniFactor, size_t TM, size_t TN, size_t TK>
void matrix_multiply(big_matrix<T1, M, N> &C, big_matrix<T2, M, K> &A,
                     big_matrix<T2, K / vnniFactor, N * vnniFactor> &B) {
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  buffer<T2, 2> bufA(A.get_data(), range<2>(M, K));
  buffer<T2, 2> bufB(B.get_data(), range<2>(K, N));
  buffer<T1, 2> bufC(C.get_data(), range<2>(M, N));

  queue q;
  q.submit([&](handler &cgh) {
     sycl::accessor accC{bufC, cgh, sycl::read_write};
     sycl::accessor accA{bufA, cgh, sycl::read_only};
     sycl::accessor accB{bufB, cgh, sycl::read_only};

     cgh.parallel_for(
         nd_range<2>({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ}),
         [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T2, use::a, TM, TK, layout::row_major> sub_a;
           joint_matrix<sub_group, T2, use::b, TK, TN, layout::ext_intel_packed>
               sub_b;
           joint_matrix<sub_group, T1, use::accumulator, TM, TN> sub_c;

           joint_matrix_load(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / SG_SZ * TN,
               N, layout::row_major);
           for (int k = 0; k < K / TK; k += 1) {
             joint_matrix_load(
                 sg, sub_a,
                 accA.template get_multi_ptr<access::decorated::no>() +
                     (sg_startx * TM) * K + k * TK,
                 K);
             joint_matrix_load(
                 sg, sub_b,
                 accB.template get_multi_ptr<access::decorated::no>() +
                     (k * TK / vnniFactor) * (N * vnniFactor) +
                     sg_starty / SG_SZ * TN * vnniFactor,
                 N * vnniFactor);
             joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
           }
         });
   }).wait();
}

template <typename Ta, typename Tc, int vnni_factor, size_t tM, size_t tN,
          size_t tK>
void init_and_multiply() {
  static constexpr size_t MATRIX_M = tM * M_MULTIPLIER;
  static constexpr size_t MATRIX_N = 128;
  static constexpr size_t MATRIX_K = 128;

  Ta A[MATRIX_M][MATRIX_K];
  Ta B[MATRIX_K][MATRIX_N];
  Ta Bvnni[MATRIX_K / vnni_factor][MATRIX_N * vnni_factor];
  Tc C[MATRIX_M][MATRIX_N];

  matrix_rand(MATRIX_M, MATRIX_K, (Ta *)A, (Ta)50);
  matrix_rand(MATRIX_K, MATRIX_N, (Ta *)B, (Ta)50);
  matrix_fill(MATRIX_M, MATRIX_N, (Tc *)C, (Tc)1);

  big_matrix<Tc, MATRIX_M, MATRIX_N> MC((Tc *)&C);
  big_matrix<Ta, MATRIX_M, MATRIX_K> MA((Ta *)&A);
  matrix_vnni<Ta>(MATRIX_K, MATRIX_N, (Ta *)&B, (Ta *)&Bvnni, vnni_factor);
  big_matrix<Ta, MATRIX_K / vnni_factor, MATRIX_N * vnni_factor> MBvnni(
      (Ta *)&Bvnni);

  matrix_multiply<Tc, Ta, MATRIX_M, MATRIX_N, MATRIX_K, vnni_factor, tM, tN,
                  tK>(MC, MA, MBvnni);
}

int main() {
  try {
    init_and_multiply<bfloat16, float, 2, 1, 500,
                      16>(); // 500 is not correct size
  } catch (const sycl::exception &e) {
    if (e.code() == errc::kernel_not_supported)
      return 0;
  }

  return 1;
}
