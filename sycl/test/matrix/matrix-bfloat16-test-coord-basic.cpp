// RUN: %clangxx -fsycl -O2 -DSYCL_EXT_ONEAPI_MATRIX_VERSION=2 %s -o %t.out
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental::matrix;
using bfloat16 = sycl::ext::oneapi::experimental::bfloat16;

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

static constexpr size_t MATRIX_M = TM * 2;
static constexpr size_t MATRIX_N = TN * 2;
static constexpr size_t MATRIX_K = TK * 2;
bfloat16 A[MATRIX_M][MATRIX_K];
bfloat16 B[MATRIX_K / 2][MATRIX_N * 2];
unsigned short Aref[MATRIX_M][MATRIX_K];
unsigned short Bref[MATRIX_K / 2][MATRIX_N * 2];
float C[MATRIX_M][MATRIX_N];
float D[MATRIX_M][MATRIX_N];
int32_t *res_local_rowA;
int32_t *res_local_colB;
int32_t *res_local_rowC;
int32_t *res_local_row_origA;
int32_t *res_local_col_origB;
int32_t *res_local_row_origC;
template <typename T1, typename T2, size_t NUM_ROWS_A, size_t NUM_COLS_A,
          size_t NUM_ROWS_B, size_t NUM_COLS_B, size_t NUM_ROWS_C,
          size_t NUM_COLS_C>
void matrix_coord(big_matrix<T1, NUM_ROWS_C, NUM_COLS_C> &C,
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

  sycl::buffer<int32_t, 1> res_local_row_bufA(res_local_rowA,
                                             sycl::range<1>(MATRIX_M));
  sycl::buffer<int32_t, 1> res_local_col_bufB(res_local_colB,
                                             sycl::range<1>(MATRIX_N));
  sycl::buffer<int32_t, 1> res_local_row_bufC(res_local_rowC,
                                             sycl::range<1>(MATRIX_M));

  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
     auto accC = bufC.get_access<sycl::access::mode::read_write>(cgh);
     auto accA = bufA.get_access<sycl::access::mode::read_write>(cgh);
     auto accB = bufB.get_access<sycl::access::mode::read_write>(cgh);

     auto res_local_row_accA =
         res_local_row_bufA.get_access<sycl::access::mode::read_write>(cgh);
     auto res_local_col_accB =
         res_local_col_bufB.get_access<sycl::access::mode::read_write>(cgh);
     auto res_local_row_accC =
         res_local_row_bufC.get_access<sycl::access::mode::read_write>(cgh);

     cgh.parallel_for<class imatrix>(
         sycl::nd_range<2>({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ}),
         [accA, accB, accC, M, N, K,
          res_local_row_accA, res_local_col_accB, res_local_row_accC](sycl::nd_item<2> spmd_item)

         {
           sycl::ext::oneapi::sub_group sg = spmd_item.get_sub_group();
           joint_matrix<bfloat16, TM, TK, use::a> sub_a(sg);
           // For B, since current implementation does not support non-packed
           // layout, users need to specify the updated VNNI sizes along with
           // the packed_b layout. By default, the layout is row_major and size
           // is (TK, TN).
           joint_matrix<bfloat16, TK, TN, use::b> sub_b(sg);
           joint_matrix<float, TM, TN, use::accumulator> sub_c(sg);

           joint_matrix_fill(sg, sub_a, 1);
           joint_matrix_fill(sg, sub_b, 2);
           joint_matrix_fill(sg, sub_c, 3); 
           // Element wise operation
           auto tAData = sub_a.get_wi_data();
           auto tBData = sub_b.get_wi_data();
           auto tCData = sub_c.get_wi_data();

           for (int i = 0; i < tAData.length(); ++i) {
             auto [row, col] = tAData[i].get_coord();
             res_local_row_accA[row] += tAData[i];
           }

           for (int i = 0; i < tBData.length(); ++i) {
             auto [row, col] = tBData[i].get_coord();
             res_local_col_accB[col] += tBData[i];
           }

           for (int i = 0; i < tCData.length(); ++i) {
             auto [row, col] = tCData[i].get_coord();
             res_local_row_accC[row] += tCData[i];
           }
         }); // parallel for
   }).wait();
}

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

void matrix_coord_ref(int *A_mem, int *B_mem, int *C_mem, int M, int N,
                         int K) { 
    for (int m = 0; m < M; m++)
      for (int k = 0; k < K; k++) {
        short *va = (short *)(A_mem + m * K + k);
        res_local_row_origA[m] += *va;
      }

    for (int k = 0; k < K; k++)
      for (int n = 0; n < N; n++) {
        short *vb = (short *)(B_mem + k * N + n);
        res_local_col_origB[n] += *vb;
      }

    for (int m = 0; m < M; m++)
      for (int n = 0; n < N; n++) {
        short *vc = (short *)(C_mem + m * N + n);
        res_local_row_origC[m] += *vc;
      }
}

int main() {
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_K; j++) {
      // Ee create bfloat16 from unsigned short since float-to-bfloat's
      // conversion is not allowed.
      A[i][j] = bfloat16::from_bits(make_bf16(1.0f));
      Aref[i][j] = make_bf16(1.0f);
    }
  }
  for (int i = 0; i < MATRIX_K / 2; i++) {
    for (int j = 0; j < MATRIX_N * 2; j++) {
      B[i][j] = bfloat16::from_bits((make_bf16(2.0f)));
      Bref[i][j] = make_bf16(2.0f);
    }
  }
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      C[i][j] = 3.0;
      D[i][j] = 3.0;
    }
  }

  big_matrix<float, MATRIX_M, MATRIX_N> MC((float *)&C);
  big_matrix<float, MATRIX_M, MATRIX_N> MD((float *)&D);
  big_matrix<bfloat16, MATRIX_M, MATRIX_K> MA((bfloat16 *)&A);
  big_matrix<bfloat16, MATRIX_K / 2, MATRIX_N * 2> MB((bfloat16 *)&B);

  res_local_rowA = (int32_t *)calloc(MATRIX_M, sizeof(int32_t));
  res_local_colB = (int32_t *)calloc(MATRIX_N, sizeof(int32_t));
  res_local_rowC = (int32_t *)calloc(MATRIX_M, sizeof(int32_t));

  res_local_row_origA = (int32_t *)calloc(MATRIX_M, sizeof(int32_t));
  res_local_col_origB = (int32_t *)calloc(MATRIX_N, sizeof(int32_t));
  res_local_row_origC = (int32_t *)calloc(MATRIX_M, sizeof(int32_t));

  matrix_coord(MC, MA, MB);
  matrix_coord_ref((int32_t *)Aref, (int32_t *)Bref, (int32_t *)D, MATRIX_M,
                      MATRIX_N, MATRIX_K / 2);

  bool res = true;
  for (int i = 0; i < MATRIX_M; i++) {
    if (res_local_rowA[i] != res_local_row_origA[i])
      res = false;
  }
  for (int i = 0; i < MATRIX_K; i++) {
    if (res_local_colB[i] != res_local_col_origB[i])
      res = false;
  }
  for (int i = 0; i < MATRIX_M; i++) {
    if (res_local_rowC[i] != res_local_row_origC[i])
      res = false;
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
