#define TM 8
#define TK 16

template <typename T1, typename T2, size_t M, size_t N, size_t K,
          unsigned int vnniFactor>
void matrix_multiply(T1 *C, T2 *A, T2 *vnniB, T2 *B, queue &q) {
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  auto pA = address_space_cast<sycl::access::address_space::global_space,
                               sycl::access::decorated::no>(A);
  auto pB = address_space_cast<sycl::access::address_space::global_space,
                               sycl::access::decorated::no>(vnniB);
  auto pC = address_space_cast<sycl::access::address_space::global_space,
                               sycl::access::decorated::no>(C);
  q.submit([&](handler &cgh) {
     cgh.parallel_for<class imatrix>(
         nd_range<2>({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ}),
         [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]]

         {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, bfloat16, use::a, TM, TK, layout::row_major>
               sub_a;
           // For B, we assume B has been already VNNIed.
           joint_matrix<sub_group, bfloat16, use::b, TK, TN,
                        layout::ext_intel_packed>
               sub_b;
           joint_matrix<sub_group, float, use::accumulator, TM, TN> sub_c;

           joint_matrix_prefetch(sg, A, (sg_startx * TM) * K, layout::row_major,
                                 TM, TK,
                                 syclex::properties{syclex::prefetch_hint_L1});
           joint_matrix_prefetch(
               sg, vnniB + sg_starty / SG_SZ * TN * vnniFactor, N * vnniFactor,
               layout::ext_intel_packed, TK / vnniFactor, TN * vnniFactor,
               syclex::properties{syclex::prefetch_hint_L1});
           joint_matrix_prefetch(sg, B + sg_starty / SG_SZ * TN, N,
                                 layout::row_major, TK, TN,
                                 syclex::properties{syclex::prefetch_hint_L2});
           joint_matrix_prefetch(
               sg, C + (sg_startx * TM) * N + sg_starty / SG_SZ * TN, N,
               layout::row_major, TM, TN,
               syclex::properties{syclex::prefetch_hint_L1});
           joint_matrix_load(sg, sub_c,
                             pC + (sg_startx * TM) * N + sg_starty / SG_SZ * TN,
                             N, layout::row_major);
           for (int k = 0; k < K / TK; k += 1) {
             joint_matrix_load(sg, sub_a, pA + (sg_startx * TM) * K + k * TK,
                               K);
             joint_matrix_load(sg, sub_b,
                               pB + (k * TK / vnniFactor) * (N * vnniFactor) +
                                   sg_starty / SG_SZ * TN * vnniFactor,
                               N * vnniFactor);
             joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
           }
           joint_matrix_store(
               sg, sub_c, pC + (sg_startx * TM) * N + sg_starty / SG_SZ * TN, N,
               layout::row_major);
         }); // parallel for
   }).wait();
}

int main() {
  queue q;
  static constexpr size_t M = TM * 2;
  static constexpr size_t N = TN * 2;
  static constexpr size_t K = TK * 2;
  static constexpr unsigned int vnniFactor = 2;
  bfloat16 *A = malloc_shared<bfloat16>(M * K, q);
  bfloat16 *B = malloc_shared<bfloat16>(K * N, q);
  bfloat16 *vnniB = malloc_shared<bfloat16>(K * N, q);
  float *C = malloc_shared<float>(M * N, q);
  float *D = malloc_shared<float>(M * N, q);

  matrix_fill(M, K, A, [](int i, int j) { return 1.0f * (i + j); });
  matrix_fill(K, N, (bfloat16 *)B,
              [](int i, int j) { return 2.0f * i + 3.0f * j; });
  matrix_fill(M, N, C, 1.0f);
  matrix_fill(M, N, D, 1.0f);

  matrix_vnni<bfloat16>(K, N, B, vnniB, vnniFactor);

  matrix_multiply<float, bfloat16, M, N, K, vnniFactor>(C, A, vnniB, B, q);
  matrix_multiply_ref(A, B, D, M, N, K);

  bool res = matrix_compare(M, N, C, D);
  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
}
