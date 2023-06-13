#define BF16_EPSILON 0.00781250
static constexpr size_t M_MULTIPLIER = 16;

template <typename T, size_t NUM_ROWS, size_t NUM_COLS> struct big_matrix {
private:
  T *mat;

public:
  T *get_data() { return mat; }
  void set_data(T *data) { mat = data; }
  big_matrix(T *data) : mat(data) {}
};

template <typename T>
void matrix_vnni(unsigned int rows, unsigned int cols, T *src, T *dest,
                 unsigned int vnniFactor) {
  for (unsigned int i = 0; i < rows / vnniFactor; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      for (unsigned int k = 0; k < vnniFactor; k++) {
        dest[i * cols * vnniFactor + j * vnniFactor + k] =
            src[(i * vnniFactor + k) * cols + j];
      }
    }
  }
}

template <typename T1, typename T2, size_t M, size_t N, size_t K,
          int vnniFactor, size_t TM, size_t TN, size_t TK>
void matrix_multiply(big_matrix<T1, M, N> &C, big_matrix<T2, M, K> &A,
                     big_matrix<T2, K / vnniFactor, N * vnniFactor> &B) {
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  buffer<T2, 2> bufA(A.get_data(), range<2>(M, K));
  buffer<T2, 2> bufB(B.get_data(), range<2>(K, N));
  buffer<T1, 2> bufC(C.get_data(), range<2>(M, N));

  queue q;
  q.submit([&](handler &cgh) {
     sycl::accessor accC{bufC, cgh, sycl::read_write};
     sycl::accessor accA{bufA, cgh, sycl::read_only};
     sycl::accessor accB{bufB, cgh, sycl::read_only};

     cgh.parallel_for(
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
           joint_matrix<sub_group, T2, use::a, TM, TK, layout::row_major> sub_a;
           // For B, we assume B has been already VNNIed.
           joint_matrix<sub_group, T2, use::b, TK, TN,
                        ext::intel::experimental::matrix::layout::packed>
               sub_b;
           joint_matrix<sub_group, T1, use::accumulator, TM, TN> sub_c;

           joint_matrix_load(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / SG_SZ * TN,
               N, layout::row_major);
           for (int k = 0; k < K / TK; k += 1) {
             joint_matrix_load(
                 sg, sub_a,
                 accA.template get_multi_ptr<access::decorated::no>() +
                     (sg_startx * TM) * K + k * TK,
                 K);
             joint_matrix_load(
                 sg, sub_b,
                 accB.template get_multi_ptr<access::decorated::no>() +
                     (k * TK / vnniFactor) * (N * vnniFactor) +
                     sg_starty / SG_SZ * TN * vnniFactor,
                 N * vnniFactor);
             sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
           }
           joint_matrix_store(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / SG_SZ * TN,
               N, layout::row_major);
         }); // parallel for
   }).wait();
}

static constexpr size_t MATRIX_N = 128;
static constexpr size_t MATRIX_K = 128;

float make_fp32(bfloat16 x) {
  unsigned int y = *((int *)&x);
  y = y << 16;
  float *res = reinterpret_cast<float *>(&y);
  return *res;
}

template <typename Ta, typename Tc>
void matrix_multiply_ref(Ta *A, Ta *B, Tc *C, int M, int N, int K) {
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        if (std::is_same_v<Ta, bfloat16> && std::is_same_v<Tc, float>)
          C[m * N + n] += make_fp32(A[m * K + k]) * make_fp32(B[k * N + n]);
        if (std::is_same_v<Ta, int8_t> && std::is_same_v<Tc, int32_t>)
          C[m * N + n] += A[m * K + k] * B[k * N + n];
      }
    }
}

template <typename Ta, typename Tc, int vnni_factor, size_t tM, size_t tN,
          size_t tK>
int init_and_multiply() {

  static constexpr size_t MATRIX_M = tM * M_MULTIPLIER;
  std::cout << "MATRIX_M=" << MATRIX_M << "\n";

  Ta A[MATRIX_M][MATRIX_K];
  Ta B[MATRIX_K][MATRIX_N];
  Ta Bvnni[MATRIX_K / vnni_factor][MATRIX_N * vnni_factor];
  Tc C[MATRIX_M][MATRIX_N];
  Tc D[MATRIX_M][MATRIX_N];

  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_K; j++) {
      if (std::is_same_v<Ta, bfloat16> && std::is_same_v<Tc, float>)
        A[i][j] = bfloat16(1.0f * (i + j));
      if (std::is_same_v<Ta, int8_t> && std::is_same_v<Tc, int32_t>)
        A[i][j] = i + j;
    }
  }
  for (int i = 0; i < MATRIX_K; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      if (std::is_same_v<Ta, bfloat16> && std::is_same_v<Tc, float>)
        B[i][j] = bfloat16(2.0f * i + 3.0f * j);
      if (std::is_same_v<Ta, int8_t> && std::is_same_v<Tc, int32_t>)
        B[i][j] = i + 2 * j;
    }
  }
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      C[i][j] = 1;
      D[i][j] = 1;
    }
  }

  big_matrix<Tc, MATRIX_M, MATRIX_N> MC((Tc *)&C);
  big_matrix<Tc, MATRIX_M, MATRIX_N> MD((Tc *)&D);
  big_matrix<Ta, MATRIX_M, MATRIX_K> MA((Ta *)&A);
  matrix_vnni<Ta>(MATRIX_K, MATRIX_N, (Ta *)&B, (Ta *)&Bvnni, vnni_factor);
  big_matrix<Ta, MATRIX_K / vnni_factor, MATRIX_N * vnni_factor> MBvnni(
      (Ta *)&Bvnni);

  matrix_multiply<Tc, Ta, MATRIX_M, MATRIX_N, MATRIX_K, vnni_factor, tM, tN,
                  tK>(MC, MA, MBvnni);
  matrix_multiply_ref((Ta *)A, (Ta *)B, (Tc *)D, MATRIX_M, MATRIX_N, MATRIX_K);

  bool res = true;
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      if constexpr (std::is_same_v<Ta, bfloat16> && std::is_same_v<Tc, float>) {
        if (fabs(C[i][j] - D[i][j]) > BF16_EPSILON) {
          res = false;
          std::cout << "Failed bfloat16: C is " << C[i][j] << ", D is "
                    << D[i][j] << std::endl;
        }
      } else if (std::is_same_v<Ta, int8_t> && std::is_same_v<Tc, int32_t>) {
        if (C[i][j] != D[i][j]) {
          res = false;
          std::cout << "Failed int8_t: C is " << C[i][j] << ", D is " << D[i][j]
                    << std::endl;
        }
      }
    }
  }
  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
}

int main() {
  int errors = 0;
  errors += init_and_multiply<bfloat16, float, 2, 1, SG_SZ, 16>();
  errors += init_and_multiply<bfloat16, float, 2, 2, SG_SZ, 16>();
  errors += init_and_multiply<bfloat16, float, 2, 3, SG_SZ, 16>();
  errors += init_and_multiply<bfloat16, float, 2, 4, SG_SZ, 16>();
  errors += init_and_multiply<bfloat16, float, 2, 5, SG_SZ, 16>();
  errors += init_and_multiply<bfloat16, float, 2, 6, SG_SZ, 16>();
  errors += init_and_multiply<bfloat16, float, 2, 7, SG_SZ, 16>();
  errors += init_and_multiply<bfloat16, float, 2, 8, SG_SZ, 16>();

  errors += init_and_multiply<int8_t, int32_t, 4, 1, SG_SZ, 32>();
  errors += init_and_multiply<int8_t, int32_t, 4, 2, SG_SZ, 32>();
  errors += init_and_multiply<int8_t, int32_t, 4, 3, SG_SZ, 32>();
  errors += init_and_multiply<int8_t, int32_t, 4, 4, SG_SZ, 32>();
  errors += init_and_multiply<int8_t, int32_t, 4, 5, SG_SZ, 32>();
  errors += init_and_multiply<int8_t, int32_t, 4, 6, SG_SZ, 32>();
  errors += init_and_multiply<int8_t, int32_t, 4, 7, SG_SZ, 32>();
  errors += init_and_multiply<int8_t, int32_t, 4, 8, SG_SZ, 32>();

  return errors;
}
