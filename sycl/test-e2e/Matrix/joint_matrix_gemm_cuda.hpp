// This tests the latest unified matrix extension interfaces.
// Specifying the sm version via the --cuda-gpu-arch flag is necessary
// for the Nvidia case.  DPC++ JIT compilation is not
// supported for the Nvidia matrix extension, although some JIT optimizations
// are performed at the level of the PTX assembly code.

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi;
using namespace sycl::ext::oneapi::experimental::matrix;
constexpr float bf16_eps = 0.00390625;

// Example usage of joint_matrix matrix multiply.
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
constexpr int SUB_TILES_N = 2;
// number of submatrices per col of "A"/per row of "B", matrices.
constexpr int SUB_TILES_K = 2;

template <typename Tm, typename Tc, typename Td, size_t M, size_t K, size_t N,
          layout layout_A, layout layout_B, layout layout_C>
class TypeHelper;

template <typename Tm, typename Tc, typename Td, size_t M, size_t K, size_t N,
          layout layout_A, layout layout_B, layout layout_C>
using KernelName =
    class TypeHelper<Tm, Tc, Td, M, K, N, layout_A, layout_B, layout_C>;

template <size_t Big_N, size_t Big_K, size_t Big_M, layout layout_A,
          layout layout_B, typename Tm, typename Tc>
Tc matrix_ref_mn(const int &m, const int &n, Tm *A, Tm *B, Tc *C) {
  Tc res = C[m * Big_N + n];

  for (int k = 0; k < Big_K; k++) {
    auto index_a =
        layout_A == layout::row_major ? m * Big_K + k : m + k * Big_M;
    auto index_b =
        layout_B == layout::row_major ? k * Big_N + n : k + n * Big_K;

    if constexpr (std::is_same<Tm, bfloat16>::value) {
      res += A[index_a] * B[index_b];
    } else {
      res += static_cast<Tc>(A[index_a]) * static_cast<Tc>(B[index_b]);
    }
  }

  return res;
}

template <
    typename Tm, typename Tc, typename Td, size_t Sub_Tiles_M,
    size_t Sub_Tiles_K, size_t Sub_Tiles_N, size_t M, size_t K, size_t N,
    layout layout_A = layout::row_major, layout layout_B = layout::row_major,
    layout layout_C = layout::row_major, typename T3 = std::remove_const_t<Tm>>
void test(queue &q) {
  // total number of M dimension matrix elements for the "Big matrix".
  constexpr auto Big_M = Sub_Tiles_M * M;
  // total number of N dimension matrix elements for the "Big matrix".
  constexpr auto Big_N = Sub_Tiles_N * N;
  // total number of K dimension matrix elements for the "Big matrix".
  constexpr auto Big_K = Sub_Tiles_K * K;

  std::remove_const_t<Tm> A[Big_M * Big_K];
  std::remove_const_t<Tm> B[Big_K * Big_N];
  std::remove_const_t<Tc> C[Big_M * Big_N];
  Td D[Big_M * Big_N];

  for (int i = 0; i < Big_M * Big_N; i++) {
    C[i] = 1;
    D[i] = 0;
  }

  if constexpr (!std::is_same<std::remove_const_t<Tm>, bfloat16>::value) {
    for (int i = 0; i < Big_M * Big_K; i++) {
      A[i] = i % 3;
    }

    for (int i = 0; i < Big_K * Big_N; i++) {
      B[i] = i % 3;
    }
  }
  {
    if constexpr (std::is_same<std::remove_const_t<Tm>, bfloat16>::value) {

      buffer<bfloat16, 1> bufA(A, range<1>(Big_M * Big_K));
      buffer<bfloat16, 1> bufB(B, range<1>(Big_K * Big_N));
      q.submit([&](handler &cgh) {
        accessor<bfloat16, 1, access::mode::write, target::device> accA(bufA,
                                                                        cgh);

        cgh.parallel_for<KernelName<Tm, Tc, class copyA, M, K, N, layout_A,
                                    layout_B, layout_C>>(
            range<1>(Big_M * Big_K), [=](item<1> item) {
              auto i = item.get_linear_id();
              accA[i] = 0.1f * (i % 10);
            });
      });
      q.submit([&](handler &cgh) {
        accessor<bfloat16, 1, access::mode::write, target::device> accB(bufB,
                                                                        cgh);

        cgh.parallel_for<KernelName<Tm, Tc, class copyB, M, K, N, layout_A,
                                    layout_B, layout_C>>(
            range<1>(Big_K * Big_N), [=](item<1> item) {
              auto i = item.get_linear_id();
              accB[i] = 0.1f * (i % 10);
            });
      });
    }

    buffer<Tm, 1> bufA(A, range<1>(Big_M * Big_K));
    buffer<Tm, 1> bufB(B, range<1>(Big_K * Big_N));
    buffer<Tc, 1> bufC(C, range<1>(Big_M * Big_N));
    buffer<Td, 1> bufD(D, range<1>(Big_M * Big_N));

    q.submit([&](handler &cgh) {
      accessor<Tm, 1, access::mode::read, target::device> accA(bufA, cgh);
      accessor<Tm, 1, access::mode::read, target::device> accB(bufB, cgh);
      accessor<Tc, 1, access::mode::read, target::device> accC(bufC, cgh);
      accessor<Td, 1, access::mode::write, target::device> accD(bufD, cgh);

      range<2> LocalRange = {1, N_THREADS_PER_MATRIX_OP};
      range<2> GlobalRange = {Sub_Tiles_M,
                              Sub_Tiles_N * N_THREADS_PER_MATRIX_OP};

      cgh.parallel_for<
          KernelName<Tm, Tc, Td, M, K, N, layout_A, layout_B, layout_C>>(
          nd_range<2>(GlobalRange, LocalRange), [=](nd_item<2> item) {
            sycl::sub_group sg = item.get_sub_group();
            // row id of current submatrix of BIG C matrix
            const auto m = item.get_group().get_group_id()[0];
            // column id of current submatrix of BIG C matrix
            const auto n = item.get_group().get_group_id()[1];

            joint_matrix<sycl::sub_group, T3, use::a, M, K, layout_A> sub_a;
            joint_matrix<sycl::sub_group, T3, use::b, K, N, layout_B> sub_b;
            joint_matrix<sycl::sub_group, std::remove_const_t<Tc>,
                         use::accumulator, M, N>
                sub_c;
            joint_matrix<sycl::sub_group, Td, use::accumulator, M, N> sub_d;
            auto stride_C = layout_C == layout::row_major ? Big_N : Big_M;
            auto load_stride_C = layout_C == layout::row_major
                                     ? (m * M) * Big_N + n * N
                                     : (m * M) + n * N * Big_M;

            joint_matrix_load(
                sg, sub_c,
                accC.template get_multi_ptr<access::decorated::no>() +
                    load_stride_C,
                stride_C, layout_C);

            auto stride_A = layout_A == layout::row_major ? Big_K : Big_M;
            auto stride_B = layout_B == layout::row_major ? Big_N : Big_K;

            // k = row/col id of current submatrix of BIG A/B matrices
            for (int k = 0; k < Sub_Tiles_K; k++) {
              auto load_stride_A = layout_A == layout::row_major
                                       ? (k * K) + (m * M * Big_K)
                                       : (k * K * Big_M) + (m * M);
              auto load_stride_B = layout_B == layout::row_major
                                       ? (k * K * Big_N) + (n * N)
                                       : (k * K) + (n * N * Big_K);

              joint_matrix_load(
                  sg, sub_a,
                  accA.template get_multi_ptr<access::decorated::no>() +
                      load_stride_A,
                  stride_A);

              joint_matrix_load(
                  sg, sub_b,
                  accB.template get_multi_ptr<access::decorated::no>() +
                      load_stride_B,
                  stride_B);

              // round values to correct precision if using tf32
              if constexpr (std::is_same<T3, precision::tf32>::value) {
                auto round_lambda = [](auto &x) { x = round_to_tf32(x); };
                joint_matrix_apply(sg, sub_a, round_lambda);
                joint_matrix_apply(sg, sub_b, round_lambda);
              }

              joint_matrix_mad(sg, sub_d, sub_a, sub_b, sub_c);
              joint_matrix_copy(sg, sub_d, sub_c);
            }
            joint_matrix_store(
                sg, sub_d,
                accD.template get_multi_ptr<access::decorated::no>() +
                    load_stride_C,
                stride_C, layout_C);
          });
    });
    q.wait();
  }

  for (int m = 0; m < Big_M; m++) {
    for (int n = 0; n < Big_N; n++) {
      auto index_D =
          layout_C == layout::row_major ? m * Big_N + n : m + n * Big_M;
      if constexpr (std::is_same<std::remove_const_t<Tm>, bfloat16>::value) {
        auto res_device =
            matrix_ref_mn<Big_N, Big_K, Big_M, layout_A, layout_B>(m, n, A, B,
                                                                   C);
        assert(fabs(2 * (D[index_D] - res_device)) / (D[index_D] + res_device) <
               bf16_eps * 2);
      } else {
        assert((D[index_D] ==
                matrix_ref_mn<Big_N, Big_K, Big_M, layout_A, layout_B>(m, n, A,
                                                                       B, C)));
      }
    }
  }
};
