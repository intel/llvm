#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel;
using namespace sycl::ext::oneapi;
using namespace sycl::ext::oneapi::experimental::matrix;

#define SG_SZ 16

#define TM 8
#define TN SG_SZ
#define TK 8

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
                     big_matrix<T2, NUM_ROWS_A, NUM_COLS_A> &A_1,
                     big_matrix<T2, NUM_ROWS_B, NUM_COLS_B> &B) {
  size_t M = NUM_ROWS_C;
  size_t N = NUM_COLS_C;
  size_t K = NUM_COLS_A;

  assert(NUM_ROWS_C == NUM_ROWS_A && NUM_COLS_A == NUM_ROWS_B);
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  buffer<float, 2> bufA(A.get_data(), range<2>(M, K));
  buffer<float, 2> bufA_1(A_1.get_data(), range<2>(M, K));

  buffer<float, 2> bufB(B.get_data(), range<2>(K, N));
  buffer<float, 2> bufC((float *)C.get_data(), range<2>(M, N));

  queue q;
  q.submit([&](handler &cgh) {
     auto accC = bufC.get_access<access::mode::read_write>(cgh);
     auto accA = bufA.get_access<access::mode::read_write>(cgh);
     auto accA_1 = bufA_1.get_access<access::mode::read_write>(cgh);
     auto accB = bufB.get_access<access::mode::read_write>(cgh);

     auto os = sycl::stream(100000, 6144, cgh);

     cgh.parallel_for<class imatrix>(
         nd_range<2>({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ}),
         [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]]

         {
           // The matrix API has to be accessed by all the workitems in a
           // subgroup these functions will be called once by the subgroup no
           // code divergence between the workitems
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, precision::tf32, use::a, TM, TK,
                        layout::row_major>
               sub_a;
           joint_matrix<sub_group, precision::tf32, use::b, TK, TN,
                        layout::row_major>
               sub_b;
           joint_matrix<sub_group, float, use::accumulator, TM, TN> sub_c;
           joint_matrix_load(sg, sub_c,
                             accC.template get_multi_ptr<access::decorated::no>() + (sg_startx * TM) * N +
                                 sg_starty / SG_SZ * TN,
                             N, layout::row_major);
           for (int k = 0; k < K; k += TK) {
            joint_matrix_load(
                 sg, sub_a, accA.template get_multi_ptr<access::decorated::no>() + (sg_startx * TM) * K + k, K);

             joint_matrix_load(
                 sg, sub_b,
                 accB.template get_multi_ptr<access::decorated::no>() + (k) * (N) + sg_starty / SG_SZ * TN, N);

             sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
           }

           joint_matrix_store(sg, sub_c,
                              accC.template get_multi_ptr<access::decorated::no>() + (sg_startx * TM) * N +
                                  sg_starty / SG_SZ * TN,
                              N, layout::row_major);
         }); // parallel for
   }).wait();
}

static constexpr size_t MATRIX_M = TM * 2;
static constexpr size_t MATRIX_N = TN * 2;
static constexpr size_t MATRIX_K = TK * 2;
float A[MATRIX_M][MATRIX_K];
float A_1[MATRIX_M][MATRIX_K];
float B[MATRIX_K][MATRIX_N];
float B_1[MATRIX_K][MATRIX_N];
float C[MATRIX_M][MATRIX_N];
float C_1[MATRIX_M][MATRIX_N];
float D[MATRIX_M][MATRIX_N];

void matrix_multiply_ref(float *A_mem, float *B_mem, float *C_mem, int M, int N,
                         int K) {
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        float va = A_mem[m * K + k];
        float vb = B_mem[k * N + n];
        C_mem[m * N + n] += va * vb;
      }
    }
}

int main() {
  std::cout << "Starting main\n";
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_K; j++) {
      A[i][j] = 1.0f; 
    }
  }
  for (int i = 0; i < MATRIX_K; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      B[i][j] = 2.0f; 
    }
  }
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      C[i][j] = 0.0; 
      D[i][j] = 0.0;
    }
  }

  big_matrix<float, MATRIX_M, MATRIX_N> MC((float *)&C);
  big_matrix<float, MATRIX_M, MATRIX_N> MD((float *)&D);
  big_matrix<float, MATRIX_M, MATRIX_K> MA((float *)&A);
  big_matrix<float, MATRIX_M, MATRIX_K> MA_1((float *)&A_1);
  big_matrix<float, MATRIX_K, MATRIX_N> MB((float *)&B);
  matrix_multiply(MC, MA, MA_1, MB);
  matrix_multiply_ref((float *)A, (float *)B, (float *)D, MATRIX_M, MATRIX_N,
                      MATRIX_K);

  bool res = true;
  std::cout << "Expected: \n";
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      std::cout << C[i][j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "**********\n";


  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      if (C[i][j] != D[i][j])
        res = false;
    }
  }

  std::cout << "Got: \n";
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      std::cout << D[i][j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
}
