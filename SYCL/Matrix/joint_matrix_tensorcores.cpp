
// REQUIRES: cuda
// Temp xfail: test was merged early.
// XFAIL: cuda
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -Xsycl-target-backend --cuda-gpu-arch=sm_80 -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4 %s -o %t.out
// RUN: %t.out
//
// This tests the latest unified matrix extension interfaces.
// Specifying the sm version via the --cuda-gpu-arch flag is necessary
// for the Nvidia case.  DPC++ JIT compilation is not
// supported for the Nvidia matrix extension, although some JIT optimizations
// are performed at the level of the PTX assembly code.

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::oneapi::experimental::matrix;
constexpr float bf16_eps = 0.00390625;

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

// the number of threads per MMA subgroup is always 32 for Nvidia.
constexpr int N_THREADS_PER_MATRIX_OP = 32;

// number of submatrices per row of accumulator ("C", "D") matrices.
constexpr int SUB_TILES_M = 2;
// number of submatrices per col of accumulator matrices.
constexpr int SUB_TILES_N = 3;
// number of submatrices per col of "A"/per row of "B", matrices.
constexpr int SUB_TILES_K = 4;

template <typename T1, typename T2, size_t M, size_t K, size_t N>
class TypeHelper;

template <typename T1, typename T2, size_t M, size_t K, size_t N>
using KernelName = class TypeHelper<T1, T2, M, K, N>;

template <size_t Big_N, size_t Big_K, typename T1, typename T2>
T2 matrix_ref_mn(const int &m, const int &n, T1 *A, T1 *B, T2 *C) {
  T2 res = C[m * Big_N + n];

  if constexpr (std::is_same<T1, bfloat16>::value) {
    for (int k = 0; k < Big_K; k++)
      res += A[m * Big_K + k] * B[k * Big_N + n];
  } else {
    for (int k = 0; k < Big_K; k++)
      res +=
          static_cast<T2>(A[m * Big_K + k]) * static_cast<T2>(B[k * Big_N + n]);
  }

  return res;
}

template <typename T1, typename T2, size_t Sub_Tiles_M, size_t Sub_Tiles_K,
          size_t Sub_Tiles_N, size_t M, size_t K, size_t N,
          typename T3 = std::remove_const_t<T1>>
void test(queue &q) {
  // total number of M dimension matrix elements for the "Big matrix".
  constexpr auto Big_M = Sub_Tiles_M * M;
  // total number of N dimension matrix elements for the "Big matrix".
  constexpr auto Big_N = Sub_Tiles_N * N;
  // total number of K dimension matrix elements for the "Big matrix".
  constexpr auto Big_K = Sub_Tiles_K * K;

  std::remove_const_t<T1> A[Big_M * Big_K];
  std::remove_const_t<T1> B[Big_K * Big_N];
  std::remove_const_t<T2> C[Big_M * Big_N];
  std::remove_const_t<T2> D[Big_M * Big_N];

  for (int i = 0; i < Big_M * Big_N; i++) {
    C[i] = 1;
    D[i] = 0;
  }

  if constexpr (!std::is_same<std::remove_const_t<T1>, bfloat16>::value) {
    for (int i = 0; i < Big_M * Big_K; i++) {
      A[i] = i % 100;
    }

    for (int i = 0; i < Big_K * Big_N; i++) {
      B[i] = i % 100;
    }
  }
  {
    if constexpr (std::is_same<std::remove_const_t<T1>, bfloat16>::value) {

      buffer<bfloat16, 1> bufA(A, range<1>(Big_M * Big_K));
      buffer<bfloat16, 1> bufB(B, range<1>(Big_K * Big_N));
      q.submit([&](handler &cgh) {
        accessor<bfloat16, 1, access::mode::write, target::device> accA(bufA,
                                                                        cgh);

        cgh.parallel_for<KernelName<T1, class copyA, M, K, N>>(
            range<1>(Big_M * Big_K), [=](item<1> item) {
              auto i = item.get_linear_id();
              accA[i] = 0.1f * (i % 10);
            });
      });
      q.submit([&](handler &cgh) {
        accessor<bfloat16, 1, access::mode::write, target::device> accB(bufB,
                                                                        cgh);

        cgh.parallel_for<KernelName<T1, class copyB, M, K, N>>(
            range<1>(Big_K * Big_N), [=](item<1> item) {
              auto i = item.get_linear_id();
              accB[i] = 0.1f * (i % 10);
            });
      });
    }

    buffer<T1, 1> bufA(A, range<1>(Big_M * Big_K));
    buffer<T1, 1> bufB(B, range<1>(Big_K * Big_N));
    buffer<T2, 1> bufC(C, range<1>(Big_M * Big_N));
    buffer<std::remove_const_t<T2>, 1> bufD(D, range<1>(Big_M * Big_N));

    q.submit([&](handler &cgh) {
      accessor<T1, 1, access::mode::read, target::device> accA(bufA, cgh);
      accessor<T1, 1, access::mode::read, target::device> accB(bufB, cgh);
      accessor<T2, 1, access::mode::read, target::device> accC(bufC, cgh);
      accessor<std::remove_const_t<T2>, 1, access::mode::write, target::device>
          accD(bufD, cgh);

      range<2> LocalRange = {1, N_THREADS_PER_MATRIX_OP};
      range<2> GlobalRange = {Sub_Tiles_M,
                              Sub_Tiles_N * N_THREADS_PER_MATRIX_OP};

      cgh.parallel_for<KernelName<T1, T2, M, K, N>>(
          nd_range<2>(GlobalRange, LocalRange), [=](nd_item<2> item) {
            sub_group sg = item.get_sub_group();
            // row id of current submatrix of BIG C matrix
            const auto m = item.get_group().get_group_id()[0];
            // column id of current submatrix of BIG C matrix
            const auto n = item.get_group().get_group_id()[1];

            joint_matrix<T3, use::a, M, K, layout::row_major> sub_a;
            joint_matrix<T3, use::b, K, N, layout::row_major> sub_b;
            joint_matrix<std::remove_const_t<T2>, use::accumulator, M, N> sub_c;

            joint_matrix_load(sg, sub_c,
                              accC.get_pointer() + (m * M) * Big_N + n * N,
                              Big_N, layout::row_major);
            // k = row/col id of current submatrix of BIG A/B matrices
            for (int k = 0; k < Sub_Tiles_K; k++) {
              joint_matrix_load(sg, sub_a,
                                accA.get_pointer() + (k * K) + (m * M * Big_K),
                                Big_K);

              joint_matrix_load(sg, sub_b,
                                accB.get_pointer() + (k * K * Big_N) + (n * N),
                                Big_N);

              // round values to correct precision if using tf32
              if constexpr (std::is_same<T3, precision::tf32>::value) {
                auto wi_size = sub_a.wi_marray.size();
                assert(wi_size == sub_b.wi_marray.size());
                for (auto i = 0; i < wi_size; ++i) {
                  sub_a.wi_marray[i] = round_to_tf32(sub_a.wi_marray[i]);
                  sub_b.wi_marray[i] = round_to_tf32(sub_b.wi_marray[i]);
                }
              }

              sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
            }
            joint_matrix_store(sg, sub_c,
                               accD.get_pointer() + (m * M) * Big_N + n * N,
                               Big_N, layout::row_major);
          });
    });
    q.wait();
  }

  for (int m = 0; m < Big_M; m++) {
    for (int n = 0; n < Big_N; n++) {
      if constexpr (std::is_same<std::remove_const_t<T1>, bfloat16>::value) {
        auto res_device = matrix_ref_mn<Big_N, Big_K>(m, n, A, B, C);
        assert(fabs(2 * (D[m * Big_N + n] - res_device)) /
                   (D[m * Big_N + n] + res_device) <
               bf16_eps * 2);
      } else {
        assert(
            (D[m * Big_N + n] == matrix_ref_mn<Big_N, Big_K>(m, n, A, B, C)));
      }
    }
  }
};

int main() {

  queue Q;
  auto computeCapability =
      std::stof(Q.get_device().get_info<sycl::info::device::backend_version>());

  if (computeCapability >= 7.0) {
    // A/B half, Accumulator float
    test<half, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 16, 16>(Q);
    test<half, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 16, 32>(Q);
    test<half, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32, 16, 8>(Q);

    test<const half, const float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 16,
         16>(Q);
    test<const half, const float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 16,
         32>(Q);
    test<const half, const float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32, 16,
         8>(Q);

    // A/B/Accumulator half
    test<half, half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 16, 16>(Q);
    test<half, half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 16, 32>(Q);
    test<half, half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32, 16, 8>(Q);

    test<const half, const half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 16,
         16>(Q);
    test<const half, const half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 16,
         32>(Q);
    test<const half, const half, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32, 16,
         8>(Q);
  }
  if (computeCapability >= 7.2) {
    test<int8_t, int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 16, 16>(Q);
    test<int8_t, int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 16, 32>(Q);
    test<int8_t, int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32, 16, 8>(Q);

    test<const int8_t, const int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16,
         16, 16>(Q);
    test<const int8_t, const int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8,
         16, 32>(Q);
    test<const int8_t, const int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32,
         16, 8>(Q);

    test<uint8_t, int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 16, 16>(
        Q);
    test<uint8_t, int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 16, 32>(Q);
    test<uint8_t, int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32, 16, 8>(Q);

    test<const uint8_t, const int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N,
         16, 16, 16>(Q);
    test<const uint8_t, const int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8,
         16, 32>(Q);
    test<const uint8_t, const int32_t, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N,
         32, 16, 8>(Q);
  }
  if (computeCapability >= 8.0) {
    test<double, double, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 4, 8>(Q);
    test<const double, const double, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8,
         4, 8>(Q);

    test<bfloat16, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 16, 16>(Q);
    test<bfloat16, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 16, 32>(Q);
    test<bfloat16, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32, 16, 8>(Q);

    test<const bfloat16, const float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16,
         16, 16>(Q);
    test<const bfloat16, const float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8,
         16, 32>(Q);
    test<const bfloat16, const float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 32,
         16, 8>(Q);

    // A/B tf32
    test<float, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 8, 16,
         precision::tf32>(Q);
    test<const float, const float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 16, 8,
         16, precision::tf32>(Q);
  }
  return 0;
};
