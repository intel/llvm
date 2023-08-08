#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bfloat16 = sycl::ext::oneapi::bfloat16;

#define TM 8
#define TN SG_SZ
#define TK 16

#define BF16_EPSILON 0.00781250

float make_fp32(bfloat16 x) {
  unsigned int y = *((int *)&x);
  y = y << 16;
  float *res = reinterpret_cast<float *>(&y);
  return *res;
}

template <typename T1, typename T2, size_t NUM_ROWS_A, size_t NUM_COLS_A,
          size_t NUM_ROWS_B, size_t NUM_COLS_B, size_t NUM_ROWS_C,
          size_t NUM_COLS_C>
void matrix_multiply(T1 *C, T2 *A, T2 *B, queue q) {
  size_t M = NUM_ROWS_C;
  size_t N = NUM_COLS_C;
  size_t K = NUM_COLS_A;

  assert(NUM_ROWS_C == NUM_ROWS_A && NUM_COLS_A == NUM_ROWS_B * 2);
  // Add one iteration for the out of bounds dpas instruction
  size_t NDRangeM = M / TM + (((M % TM) != 0) ? 1 : 0);
  size_t NDRangeN = N / TN;

  auto pA = multi_ptr<T2, sycl::access::address_space::global_space>(A);
  auto pB = multi_ptr<T2, sycl::access::address_space::global_space>(B);
  auto pC = multi_ptr<T1, sycl::access::address_space::global_space>(C);

  q.submit([&](handler &cgh) {
     cgh.parallel_for<class imatrix>(
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
           joint_matrix<sub_group, bfloat16, use::a, TM, TK, layout::row_major>
               sub_a;

           // For B, since current implementation does not support non-packed
           // layout, users need to specify the packed_b layout.
           // By default, the layout is row_major
           joint_matrix<sub_group, bfloat16, use::b, TK, TN,
                        ext::intel::experimental::matrix::layout::packed>
               sub_b;
           joint_matrix<sub_group, float, use::accumulator, TM, TN> sub_c;
           joint_matrix_fill(sg, sub_c, 1);
           for (int k = 0; k < K; k += TK) {
             joint_matrix_load(sg, sub_a, pA + (sg_startx * TM) * K + k, K);
             // Assume we alreay in vnni format.
             joint_matrix_load(
                 sg, sub_b, pB + (k) * (N) + sg_starty / SG_SZ * TN * 2, N * 2);
             sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
           }
           joint_matrix_store(
               sg, sub_c, pC + (sg_startx * TM) * N + sg_starty / SG_SZ * TN, N,
               layout::row_major);
         }); // parallel for
   }).wait();
}

void matrix_multiply_ref(bfloat16 *A, bfloat16 *B, float *C, int MATRIX_M,
                         int MATRIX_N, int MATRIX_K) {
  for (unsigned int i = 0; i < MATRIX_M; i++) {
    for (unsigned int k = 0; k < MATRIX_K; k++) {
      for (unsigned int j = 0; j < MATRIX_N; j++) {
        C[i * MATRIX_N + j] +=
            make_fp32(A[i * MATRIX_K + k]) * make_fp32(B[k * MATRIX_N + j]);
      }
    }
  }
}

template <typename T>
void matrix_vnni(unsigned int rows, unsigned int cols, T *src, T *dest,
                 unsigned int vnniFactor = 2) {
  for (unsigned int i = 0; i < rows / vnniFactor; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      for (unsigned int k = 0; k < vnniFactor; k++) {
        dest[i * cols * vnniFactor + j * vnniFactor + k] =
            src[(i * vnniFactor + k) * cols + j];
      }
    }
  }
}

int main() {
  // there will be 14*24 out of bounds in A matrix
  static constexpr size_t MATRIX_M = 1024 + 14;
  static constexpr size_t MATRIX_N = 1024;
  // K is not divisible by 8 bytes
  static constexpr size_t MATRIX_K = 1024 + 14;
  queue q;
  bfloat16 *A = malloc_shared<bfloat16>(MATRIX_M * MATRIX_K, q);
  bfloat16 *B = malloc_shared<bfloat16>(MATRIX_K * MATRIX_N, q);
  bfloat16 *vnniB = malloc_shared<bfloat16>(MATRIX_K * MATRIX_N, q);
  float *C = malloc_shared<float>(MATRIX_M * MATRIX_N, q);
  float *D = malloc_shared<float>(MATRIX_M * MATRIX_N, q);

  std::random_device dev;
  std::uniform_real_distribution<float> fdistr(-1.0, 1.0);

  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_K; j++) {
      A[i * MATRIX_K + j] = bfloat16(fdistr(dev));
    }
  }
  for (int i = 0; i < MATRIX_K; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      B[i * MATRIX_N + j] = bfloat16(fdistr(dev));
    }
  }
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      C[i * MATRIX_N + j] = 1.0;
      D[i * MATRIX_N + j] = 1.0;
    }
  }

  matrix_vnni<bfloat16>(MATRIX_K, MATRIX_N, B, vnniB, 2);
  matrix_multiply<float, bfloat16, MATRIX_M, MATRIX_K, MATRIX_K / 2,
                  MATRIX_N * 2, MATRIX_M, MATRIX_N>(C, A, vnniB, q);
  matrix_multiply_ref(A, B, D, MATRIX_M, MATRIX_N, MATRIX_K);

  bool res = true;
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      if ((fabs(C[i * MATRIX_N + j] - D[i * MATRIX_N + j])) > BF16_EPSILON) {
        std::cout << "C is " << C[i * MATRIX_N + j] << " D is "
                  << C[i * MATRIX_N + j] << " diff is "
                  << fabs(C[i * MATRIX_N + j] - D[i * MATRIX_N + j])
                  << std::endl;
        res = false;
      }
    }
  }
  if (res)
    std::cout << "passed\n";
  else
    std::cout << "failed\n";
}
