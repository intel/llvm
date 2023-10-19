#include <iostream>
#include <random>

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

  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;

  auto pA = multi_ptr<T2, sycl::access::address_space::global_space>(A);
  auto pB = multi_ptr<T2, sycl::access::address_space::global_space>(B);
  auto pC = multi_ptr<T1, sycl::access::address_space::global_space>(C);

  q.submit([&](handler &cgh) {
     cgh.parallel_for(
         nd_range<2>({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ}),
         [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]]

         {
           // The submatrix API has to be accessed by all the workitems in a
           // subgroup these functions will be called once by the subgroup no
           // code divergence between the workitems
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, float, use::accumulator, TM, TN> sub_c;
           // for transposeC
           // which TN x TM in N x M:
           //  M x N => TM x N => TM x TN => TN x TM
           //           m=sg_startx
           //                      sg_starty/SG_SZ
           // linear_index = M * (sg_starty/SG_SZ *TN) + TM *sg_startx
           joint_matrix_load(sg, sub_c,
                             pC + M * (sg_starty / SG_SZ * TN) + TM * sg_startx,
                             M, layout::col_major);
           joint_matrix_store(
               sg, sub_c, pC + M * (sg_starty / SG_SZ * TN) + TM * sg_startx, M,
               layout::col_major);
         }); // parallel for
   }).wait();
}

int main() {
  static constexpr size_t MATRIX_M = 1024;
  static constexpr size_t MATRIX_N = 1024;
  static constexpr size_t MATRIX_K = 1024;
  static constexpr unsigned int vnniFactor = 2;
  queue q;
  bfloat16 *A = malloc_shared<bfloat16>(MATRIX_M * MATRIX_K, q);
  bfloat16 *B = malloc_shared<bfloat16>(MATRIX_K * MATRIX_N, q);
  bfloat16 *vnniB = malloc_shared<bfloat16>(MATRIX_K * MATRIX_N, q);
  float *C = malloc_shared<float>(MATRIX_M * MATRIX_N, q);
  float *D = malloc_shared<float>(MATRIX_M * MATRIX_N, q);

  matrix_rand(MATRIX_M, MATRIX_K, A, (bfloat16)5);
  matrix_rand(MATRIX_K, MATRIX_N, B, (bfloat16)5);
  matrix_fill(MATRIX_M, MATRIX_N, C, (float)1.0);
  matrix_fill(MATRIX_M, MATRIX_N, D, (float)1.0);

  matrix_vnni<bfloat16>(MATRIX_K, MATRIX_N, B, vnniB, vnniFactor);
  matrix_multiply<float, bfloat16, MATRIX_M, MATRIX_K, MATRIX_K / vnniFactor,
                  MATRIX_N * vnniFactor, MATRIX_M, MATRIX_N>(C, A, vnniB, q,
                                                             vnniFactor);

  bool res = matrix_compare(MATRIX_M, MATRIX_N, C, D);

  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
}
