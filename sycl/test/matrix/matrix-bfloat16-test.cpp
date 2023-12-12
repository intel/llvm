// RUN: %clangxx -fsycl -O2 %s -o %t.out
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental::matrix;
using bfloat16 = sycl::ext::oneapi::bfloat16;

static constexpr auto TILE_SZ = 16;
static constexpr auto TM = TILE_SZ - 1;
static constexpr auto TN = TILE_SZ - 1;
static constexpr auto TK = 2 * TILE_SZ - 2;

static constexpr auto SG_SZ = 16;

template <typename T, size_t NUM_ROWS, size_t NUM_COLS> struct big_matrix {
public:
  T *mat;

public:
  T *get_data() { return mat; }
  void set_data(T *data) { mat = data; }
  big_matrix(T *data) : mat(data) {}
};

template <typename T1, typename T2, size_t NUM_ROWS_A, size_t NUM_COLS_A,
          size_t NUM_ROWS_B, size_t NUM_COLS_B, size_t NUM_ROWS_C,
          size_t NUM_COLS_C>
void matrix_multiply(big_matrix<T1, NUM_ROWS_C, NUM_COLS_C> &C,
                     big_matrix<T2, NUM_ROWS_A, NUM_COLS_A> &A,
                     big_matrix<T2, NUM_ROWS_B, NUM_COLS_B> &B) {
  size_t M = NUM_ROWS_C;
  size_t N = NUM_COLS_C;
  size_t K = NUM_COLS_A;
  // B => K/4 x N*4, A => M x K, C => M, N
  // stride should be X's cols, e.g., B's stirde = N*4
  assert(NUM_ROWS_C == NUM_ROWS_A && NUM_COLS_A == NUM_ROWS_B * 2);
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  sycl::buffer<bfloat16, 2> bufA(A.get_data(), sycl::range<2>(M, K));
  sycl::buffer<bfloat16, 2> bufB(B.get_data(), sycl::range<2>(K, N));
  sycl::buffer<float, 2> bufC((float *)C.get_data(), sycl::range<2>(M, N));

  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
     auto accC = bufC.get_access<sycl::access::mode::read_write>(cgh);
     auto accA = bufA.get_access<sycl::access::mode::read_write>(cgh);
     auto accB = bufB.get_access<sycl::access::mode::read_write>(cgh);

     cgh.parallel_for<class imatrix>(
         sycl::nd_range<2>({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ}),
         [accA, accB, accC, M, N, K](sycl::nd_item<2> spmd_item)

         {
           // The submatrix API has to be accessed by all the workitems in a
           // subgroup these functions will be called once by the subgroup no
           // code divergence between the workitems
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           joint_matrix<sycl::sub_group, bfloat16, use::a, TM, TK,
                        layout::row_major>
               sub_a;
           // For B, since current implementation does not support non-packed
           // layout, users need to specify the updated VNNI sizes along with
           // the packed_b layout. By default, the layout is row_major and size
           // is (TK, TN).
           joint_matrix<sycl::sub_group, bfloat16, use::b, TK, TN,
                        layout::ext_intel_packed>
               sub_b;
           joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> sub_c;

           joint_matrix_load(
               spmd_item.get_sub_group(), sub_c,
               accC.template get_multi_ptr<sycl::access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / SG_SZ * TN,
               N, layout::row_major);
           for (int k = 0; k < K / TK; k += 1) { //
             joint_matrix_load(
                 spmd_item.get_sub_group(), sub_a,
                 accA.template get_multi_ptr<sycl::access::decorated::no>() +
                     (sg_startx * TM) * K + k * TK,
                 K);
             // Assuming B data is already in VNNI format.
             joint_matrix_load(
                 spmd_item.get_sub_group(), sub_b,
                 accB.template get_multi_ptr<sycl::access::decorated::no>() +
                     (k * TK / 2) * (N * 2) + sg_starty / SG_SZ * TN * 2,
                 N * 2);
             joint_matrix_mad(spmd_item.get_sub_group(), sub_c, sub_a, sub_b,
                              sub_c);
           }
           joint_matrix_store(
               spmd_item.get_sub_group(), sub_c,
               accC.template get_multi_ptr<sycl::access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / SG_SZ * TN,
               N, layout::row_major);
         }); // parallel for
   }).wait();
}

static constexpr size_t MATRIX_M = TM * 2;
static constexpr size_t MATRIX_N = TN * 2;
static constexpr size_t MATRIX_K = TK * 2;
bfloat16 A[MATRIX_M][MATRIX_K];
bfloat16 B[MATRIX_K / 2][MATRIX_N * 2];
unsigned short Aref[MATRIX_M][MATRIX_K];
unsigned short Bref[MATRIX_K / 2][MATRIX_N * 2];
float C[MATRIX_M][MATRIX_N];
float D[MATRIX_M][MATRIX_N];

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

void matrix_multiply_ref(int *A_mem, int *B_mem, int *C_mem, int M, int N,
                         int K) {
  // tiling
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        short *va = (short *)(A_mem + m * K + k);
        short *vb = (short *)(B_mem + k * N + n);
        float acc = *((float *)(C_mem + m * N + n));
        // FIXME: Should we do reduce-add in another version?
        for (int i = 0; i < 2; i++) {
          acc += (make_fp32(va[i]) * make_fp32(vb[i]));
        }
        *((float *)(C_mem + m * N + n)) = acc;
      }
    }
}

int main() {
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_K; j++) {
      // Ee create bfloat16 from unsigned short since float-to-bfloat's
      // conversion is not allowed.
      A[i][j] = bfloat16(1.0f * (i + j));
      Aref[i][j] = make_bf16(1.0f * (i + j));
    }
  }
  for (int i = 0; i < MATRIX_K / 2; i++) {
    for (int j = 0; j < MATRIX_N * 2; j++) {
      B[i][j] = bfloat16(2.0f * i + 3.0f * j);
      Bref[i][j] = make_bf16(2.0f * i + 3.0f * j);
    }
  }
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      C[i][j] = 1.0;
      D[i][j] = 1.0;
    }
  }

  big_matrix<float, MATRIX_M, MATRIX_N> MC((float *)&C);
  big_matrix<float, MATRIX_M, MATRIX_N> MD((float *)&D);
  big_matrix<bfloat16, MATRIX_M, MATRIX_K> MA((bfloat16 *)&A);
  big_matrix<bfloat16, MATRIX_K / 2, MATRIX_N * 2> MB((bfloat16 *)&B);
  matrix_multiply(MC, MA, MB);
  matrix_multiply_ref((int32_t *)Aref, (int32_t *)Bref, (int32_t *)D, MATRIX_M,
                      MATRIX_N, MATRIX_K / 2);

  bool res = true;
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      if (C[i][j] != D[i][j])
        res = false;
    }
  }
  if (res)
    std::cout << "passed\n";
  else
    std::cout << "failed\n";
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++)
      std::cout << C[i][j] << ", ";
    std::cout << "\n";
  }
  std::cout << std::endl;
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++)
      std::cout << D[i][j] << ", ";
    std::cout << "\n";
  }
}
