#include <iostream>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

constexpr size_t TM = 8;
constexpr size_t TK = 16;

template <typename T1, typename T2, size_t NUM_ROWS_A, size_t NUM_COLS_A,
          size_t NUM_ROWS_B, size_t NUM_COLS_B, size_t NUM_ROWS_C,
          size_t NUM_COLS_C>
void matrix_multiply(T1 *C, T2 *A, T2 *B, queue q, unsigned int vnniFactor) {
  size_t M = NUM_ROWS_C;
  size_t N = NUM_COLS_C;
  size_t K = NUM_COLS_A;

  assert(NUM_ROWS_C == NUM_ROWS_A && NUM_COLS_A == NUM_ROWS_B * vnniFactor);
  // Add one iteration for the out of bounds dpas instruction
  size_t NDRangeM = M / TM + (((M % TM) != 0) ? 1 : 0);
  size_t NDRangeN = N / TN;
  q.submit([&](handler &cgh) {
     cgh.parallel_for(
         nd_range<2>({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ}),
         [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]]

         {
           auto pA =
               address_space_cast<sycl::access::address_space::global_space,
                                  sycl::access::decorated::no>(A);
           auto pB =
               address_space_cast<sycl::access::address_space::global_space,
                                  sycl::access::decorated::no>(B);
           auto pC =
               address_space_cast<sycl::access::address_space::global_space,
                                  sycl::access::decorated::no>(C);
           // The submatrix API has to be accessed by all the workitems in a
           // subgroup these functions will be called once by the subgroup no
           // code divergence between the workitems
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, bfloat16, use::a, TM, TK, layout::row_major>
               sub_a;

           // For B, since current implementation does not support non-packed
           // layout, users need to specify the packed_b layout.
           joint_matrix<sub_group, bfloat16, use::b, TK, TN,
                        layout::ext_intel_packed>
               sub_b;
           joint_matrix<sub_group, float, use::accumulator, TM, TN> sub_c;
           // bounds-checked load where width and height are added
           ext::intel::experimental::matrix::joint_matrix_fill_checked(
               sg, sub_c, 1, N, M, N, sg_startx * TM, sg_starty / SG_SZ * TN);
           for (int k = 0; k < K; k += TK) {
             // bounds-checked load where width and height are added
             ext::intel::experimental::matrix::joint_matrix_load_checked(
                 sg, sub_a, pA, K, M, K, sg_startx * TM, k);
             // Assume we alreay in vnni format.
             // bounds-checked load where width and height are added
             ext::intel::experimental::matrix::joint_matrix_load_checked(
                 sg, sub_b, pB, N * vnniFactor, K / vnniFactor, N * vnniFactor,
                 k, sg_starty / SG_SZ * TN * vnniFactor);
             joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
           }
           // bounds-checked store where width and height are added
           ext::intel::experimental::matrix::joint_matrix_store_checked(
               sg, sub_c, pC, N, layout::row_major, M, N, sg_startx * TM,
               sg_starty / SG_SZ * TN);
         }); // parallel for
   }).wait();
}

int main() {
  static constexpr size_t MATRIX_M = 1024 + 14;
  static constexpr size_t MATRIX_N = 1024;
  static constexpr unsigned int vnniFactor = 2;

  queue q;
  bfloat16 *A = malloc_shared<bfloat16>(MATRIX_M * MATRIX_K, q);
  bfloat16 *B = malloc_shared<bfloat16>(MATRIX_K * MATRIX_N, q);
  bfloat16 *vnniB = malloc_shared<bfloat16>(MATRIX_K * MATRIX_N, q);
  float *C = malloc_shared<float>(MATRIX_M * MATRIX_N, q);
  float *D = malloc_shared<float>(MATRIX_M * MATRIX_N, q);

  matrix_rand(MATRIX_M, MATRIX_K, A, (bfloat16)5);
  matrix_rand(MATRIX_K, MATRIX_N, B, (bfloat16)5);
  matrix_fill(MATRIX_M, MATRIX_N, C, (float)1);
  matrix_fill(MATRIX_M, MATRIX_N, D, (float)1);

  matrix_vnni<bfloat16>(MATRIX_K, MATRIX_N, B, vnniB, vnniFactor);
  matrix_multiply<float, bfloat16, MATRIX_M, MATRIX_K, MATRIX_K / vnniFactor,
                  MATRIX_N * vnniFactor, MATRIX_M, MATRIX_N>(C, A, vnniB, q,
                                                             vnniFactor);
  matrix_multiply_ref(A, B, D, MATRIX_M, MATRIX_N, MATRIX_K);

  bool res = matrix_compare(MATRIX_M, MATRIX_N, C, D);

  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
}
