// REQUIRES: gpu, cuda

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -Xsycl-target-backend --cuda-gpu-arch=sm_80 -DSYCL_EXT_ONEAPI_MATRIX=3  %s -o %t.out
//
// Specifying the sm version via the --cuda-gpu-arch flag is necessary
// for the Nvidia case.  DPC++ JIT compilation is not
// supported for the Nvidia matrix extension, although some JIT optimizations
// are performed at the level of the PTX assembly code.

#include <CL/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

// Example usage of Nvidia matrix multiply.
// Optimizations such as memory paddings for avoiding bank conflicts are not
// included in this test which aids clarity for what is going on. This example
// forms a "Big matrix" corresponding to a single "TILE" using cuda example
// terminology.  Multiple TILES can be used to construct yet larger matrices.
// This example uses row_major a, b, and accumulator matrices.

// M, N, K define the unit sizes of dimensions of the three types (a, b,
// accumulator) of matrices per subgroup operation:
// M: number of rows of "C"/"D" (Accumulator) sub-matrices,
// number of cols of "B" sub-matrix.
// N: number of cols of "C"/"D" (Accumulator) sub-matrices,
// number of rows of "A" sub-matrix.
// K: number of cols of "A"/number of rows of "B" sub-matrices.

constexpr int N_THREADS_PER_MATRIX_OP =
    32; // the number of threads per MMA subgroup is always 32 for Nvidia.

constexpr int SUB_TILES_M =
    2; // number of submatrices per row of accumulator ("C", "D") matrices.
constexpr int SUB_TILES_N =
    3; // number of submatrices per col of accumulator matrices.
constexpr int SUB_TILES_K =
    4; // number of submatrices per col of "A"/per row of "B", matrices.

template <typename T1, typename T2, size_t M, size_t K, size_t N>
class TypeHelper;

template <typename T1, typename T2, size_t M, size_t K, size_t N>
using KernelName = class TypeHelper<T1, T2, M, K, N>;

float make_fp32(short x) {
  unsigned int y = x;
  y = y << 16;
  float *res = reinterpret_cast<float *>(&y);
  return *res;
}

unsigned short make_bf16(float x) {
  int *res = reinterpret_cast<int *>(&x);
  *res = *res >> 16;
  return (unsigned short)*res;
}

template <typename T1, typename T2, size_t Big_N, size_t Big_K>
T2 matrix_ref_mn(const int &m, const int &n, T1 *A, T1 *B, T2 *C) {
  T2 res = C[m * Big_N + n];

  if constexpr (std::is_same<T1, uint16_t>::value) {
    for (int k = 0; k < Big_K; k++)
      res += make_fp32(A[m * Big_K + k]) * make_fp32(B[k * Big_N + n]);
  } else {
    for (int k = 0; k < Big_K; k++)

      res +=
          static_cast<T2>(A[m * Big_K + k]) * static_cast<T2>(B[k * Big_N + n]);
  }

  return res;
}

template <typename T1, typename T2, size_t Sub_Tiles_M, size_t Sub_Tiles_K,
          size_t Sub_Tiles_N, size_t M, size_t K, size_t N>
void test() {

  constexpr auto Big_M =
      Sub_Tiles_M *
      M; // total number of M dimension matrix elements for the "Big matrix".
  constexpr auto Big_N =
      Sub_Tiles_N *
      N; // total number of N dimension matrix elements for the "Big matrix".
  constexpr auto Big_K =
      Sub_Tiles_K *
      K; // total number of K dimension matrix elements for the "Big matrix".

  T1 A[Big_M * Big_K];
  T1 B[Big_K * Big_N];
  T2 C[Big_M * Big_N];
  T2 D[Big_M * Big_N];

  for (int i = 0; i < Big_M * Big_N; i++) {
    C[i] = 1;
    D[i] = 0;
  }

  if constexpr (std::is_same<T1, uint16_t>::value) {
    for (int i = 0; i < Big_M * Big_K; i++) {
      A[i] = make_bf16(0.1f * (i % 10));
    }

    for (int i = 0; i < Big_K * Big_N; i++) {
      B[i] = make_bf16(0.1f * (i % 10));
    }
  } else {
    for (int i = 0; i < Big_M * Big_K; i++) {
      A[i] = i % 100;
    }

    for (int i = 0; i < Big_K * Big_N; i++) {
      B[i] = i % 100;
    }
  }

  buffer<T1, 1> bufA(A, range<1>(Big_M * Big_K));
  buffer<T1, 1> bufB(B, range<1>(Big_K * Big_N));
  buffer<T2, 1> bufC(C, range<1>(Big_M * Big_N));
  buffer<T2, 1> bufD(D, range<1>(Big_M * Big_N));

  queue q;
  q.submit([&](handler &cgh) {
    auto accC = bufC.template get_access<access::mode::read_write>(cgh);
    auto accA = bufA.template get_access<access::mode::read_write>(cgh);
    auto accB = bufB.template get_access<access::mode::read_write>(cgh);
    auto accD = bufD.template get_access<access::mode::read_write>(cgh);

    range<2> LocalRange = {1, N_THREADS_PER_MATRIX_OP};
    range<2> GlobalRange = {Sub_Tiles_M, Sub_Tiles_N * N_THREADS_PER_MATRIX_OP};

    cgh.parallel_for<KernelName<T1, T2, M, K, N>>(
        nd_range<2>(GlobalRange, LocalRange), [=
    ](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();
          const auto m =
              item.get_group()
                  .get_id()[0]; // row id of current submatrix of BIG C matrix
          const auto n =
              item.get_group().get_id()[1]; // column id of current
                                            // submatrix of BIG C matrix

          joint_matrix<T1, matrix_use::a, M, K, matrix_layout::row_major> sub_a;

          joint_matrix<T1, matrix_use::b, K, N, matrix_layout::row_major> sub_b;

          joint_matrix<T2, matrix_use::accumulator, M, N,
                       matrix_layout::row_major>
              sub_c;

          joint_matrix_load(
              sg, sub_c, accC.get_pointer() + (m * M) * Big_N + n * N, Big_N);

          for (int k = 0; k < Sub_Tiles_K;
               k++) // row/col id of current submatrix of BIG A/B matrices
          {
            joint_matrix_load(sg, sub_a,
                              accA.get_pointer() + (k * K) + (m * M * Big_K),
                              Big_K);

            joint_matrix_load(sg, sub_b,
                              accB.get_pointer() + (k * K * Big_N) + (n * N),
                              Big_N);

            sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          }
          joint_matrix_store(
              sg, sub_c, accD.get_pointer() + (m * M) * Big_N + n * N, Big_N);
        });
  });

  q.wait();

  const auto host_accessor = bufD.template get_access<access::mode::read>();
  for (int m = 0; m < Big_M; m++)
    for (int n = 0; n < Big_N; n++) {

      assert((host_accessor[m * Big_N + n] ==
              matrix_ref_mn<T1, T2, Big_N, Big_K>(m, n, A, B, C)));
    }
};

int main() {

  // A/B half, Accumulator float
  test<half, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 16, 16>();
  test<half, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 16, 32>();
  test<half, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32, 16, 8>();

  // A/B/Accumulator half
  test<half, half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 16, 16>();
  test<half, half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 16, 32>();
  test<half, half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32, 16, 8>();

  test<int8_t, int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 16, 16>();
  test<int8_t, int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 16, 32>();
  test<int8_t, int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32, 16, 8>();

  test<uint8_t, int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 16, 16>();
  test<uint8_t, int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 16, 32>();
  test<uint8_t, int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32, 16, 8>();

  test<double, double, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 4, 8>();

  // A/B bf16
  test<uint16_t, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 16, 16>();
  test<uint16_t, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 16, 32>();
  test<uint16_t, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32, 16, 8>();

  return 0;
};
