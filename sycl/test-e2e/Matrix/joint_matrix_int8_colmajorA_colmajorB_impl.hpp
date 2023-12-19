#define TM 8
#define TK 32

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
           joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::col_major>
               sub_a;
           joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::col_major>
               sub_b;
           joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> sub_c;

           joint_matrix_fill(sg, sub_c, 0);
           for (int k = 0; k < K / TK; k += 1) {
             joint_matrix_load(
                 sg, sub_a,
                 accA.template get_multi_ptr<access::decorated::no>() +
                     (k * TK) * M + sg_startx * TM,
                 M);
             joint_matrix_load(
                 sg, sub_b,
                 accB.template get_multi_ptr<access::decorated::no>() +
                     (sg_starty / SG_SZ * TN) * K + k * TK,
                 K);
             joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
           }
           joint_matrix_store(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / SG_SZ * TN,
               N, layout::row_major);
         }); // parallel for
   }).wait();
}

int main() {
  static constexpr size_t MATRIX_M = TM;
  static constexpr size_t MATRIX_N = TN;
  static constexpr size_t MATRIX_K = TK;
  int8_t A[MATRIX_K][MATRIX_M];
  int8_t Aref[MATRIX_K][MATRIX_M];
  int8_t B[MATRIX_N][MATRIX_K];
  int8_t Bref[MATRIX_N][MATRIX_K];
  int32_t C[MATRIX_M][MATRIX_N];
  int32_t D[MATRIX_M][MATRIX_N];

  matrix_fill(MATRIX_K, MATRIX_M, (int8_t *)A,
              [](int i, int j) { return 2 * i + j; });
  matrix_fill(MATRIX_K, MATRIX_M, (int8_t *)Aref,
              [](int i, int j) { return 2 * i + j; });

  matrix_fill(MATRIX_N, MATRIX_K, (int8_t *)B,
              [](int i, int j) { return i + 2 * j; });
  matrix_fill(MATRIX_N, MATRIX_K, (int8_t *)Bref,
              [](int i, int j) { return i + 2 * j; });

  matrix_fill(MATRIX_M, MATRIX_N, (int32_t *)C, 0);
  matrix_fill(MATRIX_M, MATRIX_N, (int32_t *)D, 0);

  big_matrix<int32_t, MATRIX_M, MATRIX_N> MC((int32_t *)&C);
  big_matrix<int32_t, MATRIX_M, MATRIX_N> MD((int32_t *)&D);
  big_matrix<int8_t, MATRIX_M, MATRIX_K> MA((int8_t *)&A);
  big_matrix<int8_t, MATRIX_K, MATRIX_N> MB((int8_t *)&B);
  matrix_multiply(MC, MA, MB);
  matrix_multiply_ref((int8_t *)Aref, (int8_t *)Bref, (int32_t *)D, MATRIX_M,
                      MATRIX_N, MATRIX_K, false, true, true);

  bool res = matrix_compare(MATRIX_M, MATRIX_N, (int32_t *)C, (int32_t *)D);
  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
}
