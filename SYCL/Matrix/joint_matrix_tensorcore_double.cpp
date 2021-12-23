// The test is disabled to devil's issue #666
// REQUIRES: gpu, cuda, TEMPORARY_DISABLED

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -Xsycl-target-backend --cuda-gpu-arch=sm_80 -DSYCL_EXT_ONEAPI_MATRIX=3  %s -o %t.out
//
// Specifying the sm version via the --cuda-gpu-arch flag is necessary
// for the Nvidia case.  DPC++ JIT compilation is not
// supported for the Nvidia case, although some JIT optimizations are performed
// at the level of the PTX assembly code.

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
// accumulator) of matrices per subgroup operation.
constexpr int M = 8; // number of rows of "C"/"D" (Accumulator) sub-matrices,
                     // number of cols of "B" sub-matrix.
constexpr int N = 8; // number of cols of "C"/"D" (Accumulator) sub-matrices,
                     // number of rows of "A" sub-matrix.
constexpr int K =
    4; // number of cols of "A"/number of rows of "B" sub-matrices.

constexpr int N_THREADS_PER_MATRIX_OP =
    32; // the number of threads per MMA subgroup is always 32 for Nvidia.

constexpr int SUB_TILES_M =
    77; // number of submatrices per row of accumulator ("C", "D") matrices.
constexpr int SUB_TILES_N =
    55; // number of submatrices per col of accumulator matrices.
constexpr int SUB_TILES_K =
    257; // number of submatrices per col of "A"/per row of "B", matrices.

constexpr int BIG_M =
    SUB_TILES_M *
    M; // total number of M dimension matrix elements for the "Big matrix".
constexpr int BIG_N =
    SUB_TILES_N *
    N; // total number of N dimension matrix elements for the "Big matrix".
constexpr int BIG_K =
    SUB_TILES_K *
    K; // total number of K dimension matrix elements for the "Big matrix".

// The stride should equal the number of elements between consecutive leading
// dimensions of the "Big matrix". e.g. number of elements per row if matrix is
// indexed row major. The stride tells the implementation how many elements to
// skip in memory matrix row/column multiplications.
constexpr int STRIDE_A = BIG_K; // row major. If col major should equal BIG_M.
constexpr int STRIDE_B = BIG_N; // row_major. If col major should equal BIG_K.
constexpr int STRIDE_C = BIG_N; // row major. If col major should equal BIG_M.

double A[BIG_M * BIG_K];
double B[BIG_K * BIG_N];
double C[BIG_M * BIG_N];
double D[BIG_M * BIG_N];

// returns correct (m,n) element of matrix D = A*B + C (assuming all matrices
// are indexed row_major).
double matrix_ref_mn(const int &m, const int &n) {
  double res = C[m * BIG_N + n];

  for (int k = 0; k < BIG_K; k++)
    res += A[m * BIG_K + k] * B[k * BIG_N + n];
  return res;
}

int main() {
  for (int i = 0; i < BIG_M * BIG_N; i++) {
    C[i] = i;
    D[i] = 0;
  }

  for (int i = 0; i < BIG_M * BIG_K; i++) {
    A[i] = i;
  }

  for (int i = 0; i < BIG_K * BIG_N; i++) {
    B[i] = i;
  }

  buffer<double, 1> bufA(A, range<1>(BIG_M * BIG_K));
  buffer<double, 1> bufB(B, range<1>(BIG_K * BIG_N));
  buffer<double, 1> bufC(C, range<1>(BIG_M * BIG_N));
  buffer<double, 1> bufD(D, range<1>(BIG_M * BIG_N));

  queue q;
  q.submit([&](handler &cgh) {
    auto accC = bufC.get_access<access::mode::read_write>(cgh);
    auto accA = bufA.get_access<access::mode::read_write>(cgh);
    auto accB = bufB.get_access<access::mode::read_write>(cgh);
    auto accD = bufD.get_access<access::mode::read_write>(cgh);

    range<2> LocalRange = {1, N_THREADS_PER_MATRIX_OP};
    range<2> GlobalRange = {SUB_TILES_M, SUB_TILES_N * N_THREADS_PER_MATRIX_OP};

    cgh.parallel_for<class imatrix>(
        nd_range<2>(GlobalRange, LocalRange), [=
    ](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          const auto m =
              item.get_group()
                  .get_id()[0]; // row id of current submatrix of BIG C matrix
          const auto n =
              item.get_group().get_id()[1]; // column id of current submatrix of
                                            // BIG C matrix

          joint_matrix<double, matrix_use::accumulator, M, N,
                       matrix_layout::row_major>
              sub_c;

          joint_matrix<double, matrix_use::a, M, K, matrix_layout::row_major>
              sub_a;

          joint_matrix<double, matrix_use::b, K, N, matrix_layout::row_major>
              sub_b;

          joint_matrix_load(sg, sub_c,
                            accC.get_pointer() + (m * M) * BIG_N + n * N,
                            STRIDE_C);

          for (int k = 0; k < SUB_TILES_K;
               k += 1) // row/col id of current submatrix of BIG A/B matrices
          {
            joint_matrix_load(sg, sub_a,
                              accA.get_pointer() + (k * K) + (m * M * BIG_K),
                              STRIDE_A);

            joint_matrix_load(sg, sub_b,
                              accB.get_pointer() + (k * K * BIG_N) + (n * N),
                              STRIDE_B);

            sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          }
          joint_matrix_store(sg, sub_c,
                             accD.get_pointer() + (m * M) * BIG_N + n * N,
                             STRIDE_C);
        });
  });

  const auto host_accessor = bufD.get_access<cl::sycl::access::mode::read>();

  for (int m = 0; m < BIG_M; m++)
    for (int n = 0; n < BIG_N; n++) {
      assert(host_accessor[m * BIG_N + n] == matrix_ref_mn(m, n));
    }

  return 0;
};
