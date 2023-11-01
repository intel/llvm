#include <iostream>
#include <random>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

constexpr size_t TM = 8;
constexpr size_t TK = 16;

template <typename T1, size_t NUM_ROWS_C, size_t NUM_COLS_C>
void matrix_load_store(T1 *C, queue q) {
  size_t M = NUM_ROWS_C;
  size_t N = NUM_COLS_C;

  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;

  auto pC = address_space_cast<sycl::access::address_space::global_space,
                               sycl::access::decorated::no>(C);

  q.submit([&](handler &cgh) {
     cgh.parallel_for(
         nd_range<2>({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ}),
         [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]]

         {
           // The submatrix API has to be accessed by all the workitems in
           // a subgroup these functions will be called once by the
           // subgroup no code divergence between the workitems
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

  queue q;
  float *C = malloc_shared<float>(MATRIX_M * MATRIX_N, q);
  float *D = malloc_shared<float>(MATRIX_M * MATRIX_N, q);

  matrix_rand(MATRIX_M, MATRIX_N, C, (float)5.0);
  matrix_copy(MATRIX_M, MATRIX_N, C, D);

  matrix_load_store<float, MATRIX_M, MATRIX_N>(C, q);

  bool res = matrix_compare(MATRIX_M, MATRIX_N, C, D);

  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
}
