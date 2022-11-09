#define TM 8
#define TN SG_SZ
#define TK 32

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

  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  buffer<int8_t, 2> bufA(A.get_data(), range<2>(M, K));
  buffer<int8_t, 2> bufB(B.get_data(), range<2>(K, N));
  buffer<int32_t, 2> bufC(C.get_data(), range<2>(M, N));

  queue q;
  q.submit([&](handler &cgh) {
     auto accC = bufC.get_access<access::mode::read_write>(cgh);
     auto accA = bufA.get_access<access::mode::read_write>(cgh);
     auto accB = bufB.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for<class imatrix>(
         nd_range<2>({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ}),
         [accA, accB, accC, M, N, K](nd_item<2> spmd_item)

         {
           // The submatrix API has to be accessed by all the workitems in a
           // subgroup these functions will be called once by the subgroup no
           // code divergence between the workitems
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<int8_t, TM, TK> sub_a(sg);
           joint_matrix<int8_t, TK, TN, matrix_layout::packed_b> sub_b(sg);
           joint_matrix<int32_t, TM, TN> sub_c(sg);

           joint_matrix_fill(sg, sub_c, 0);
           for (int k = 0; k < K / TK; k += 1) {
             joint_matrix_load(
                 sg, sub_a, accA.get_pointer() + (sg_startx * TM) * K + k * TK,
                 K, matrix_layout::row_major);
             // VNNI transform is done automatically at this level
             joint_matrix_load(sg, sub_b,
                               accB.get_pointer() + (k * TK) * N +
                                   sg_starty / SG_SZ * TN,
                               N, matrix_layout::row_major);
             sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
           }
           joint_matrix_store(sg, sub_c,
                              accC.get_pointer() + (sg_startx * TM) * N +
                                  sg_starty / SG_SZ * TN,
                              N, matrix_layout::row_major);
         }); // parallel for
   }).wait();
}

static constexpr size_t MATRIX_M = TM * 2;
static constexpr size_t MATRIX_N = TN * 2;
static constexpr size_t MATRIX_K = TK * 2;
int8_t A[MATRIX_M][MATRIX_K];
int8_t B[MATRIX_K][MATRIX_N];
int8_t Bvnni[MATRIX_K / 4][MATRIX_N * 4];
int32_t C[MATRIX_M][MATRIX_N];
int32_t D[MATRIX_M][MATRIX_N];

void int8_row_vnni_reformat(int8_t *_in, int8_t *_out, int K, int N,
                            int stride_in) {
  // find the old index, new index, and copy element.
  //(K, N) => (k/4, N*4)
  // idx in 2d: (i,j)=>(i/4, j*4+i%4)
  // linear idx:
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N; ++j) {
      size_t oldindex = i * stride_in + j;
      size_t newindex = (i / 4) * N * 4 + j * 4 + i % 4;
      _out[newindex] = _in[oldindex];
    }
  }
}

void matrix_multiply_ref(int32_t *A_mem, int32_t *B_mem, int32_t *C_mem, int M,
                         int N, int K) {
  // tiling
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        char *va = (char *)(A_mem + m * K + k);
        char *vb = (char *)(B_mem + k * N + n);
        int acc = *(C_mem + m * N + n);
        for (int i = 0; i < 4; i++) {
          acc += (va[i] * vb[i]);
        }
        *(C_mem + m * N + n) = acc;
      }
    }
}

int main() {
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_K; j++) {
      A[i][j] = i + j;
    }
  }
  for (int i = 0; i < MATRIX_K; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      B[i][j] = i + j * 2;
    }
  }
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      C[i][j] = 0;
      D[i][j] = 0;
    }
  }

  big_matrix<int32_t, MATRIX_M, MATRIX_N> MC((int32_t *)&C);
  big_matrix<int32_t, MATRIX_M, MATRIX_N> MD((int32_t *)&D);
  big_matrix<int8_t, MATRIX_M, MATRIX_K> MA((int8_t *)&A);
  big_matrix<int8_t, MATRIX_K / 4, MATRIX_N * 4> MB((int8_t *)&B);
  matrix_multiply(MC, MA, MB);
  int8_row_vnni_reformat((int8_t *)B, (int8_t *)Bvnni, MATRIX_K, MATRIX_N,
                         MATRIX_N);
  matrix_multiply_ref((int32_t *)A, (int32_t *)Bvnni, (int32_t *)D, MATRIX_M,
                      MATRIX_N, MATRIX_K / 4);

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
}
