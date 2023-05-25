#define TM 8
#define TN SG_SZ
#define TK 16

static constexpr size_t MATRIX_M = TM * 2;
static constexpr size_t MATRIX_N = TN * 2;
static constexpr size_t MATRIX_K = TK * 2;

#define BF16_EPSILON 0.00781250

template <typename T, size_t NUM_ROWS, size_t NUM_COLS> struct big_matrix {
private:
  T *mat;

public:
  T *get_data() { return mat; }
  void set_data(T *data) { mat = data; }
  big_matrix(T *data) : mat(data) {}
};

// clang-format off
/*
Here's how the data is distributed
W0 --> 0 1 2 3 4 5 6 7
wi [0,0] -> i=0, [0, 0]        wi [0,1] --> i=0, [0, 1]     wi [0,15] --> i=0, [0, 15]
            i=1, [1, 0]                     i=1, [1, 1]                   i=1, [1, 15]
            i=2, [2, 0]                     i=2, [2, 1]                   ...
            ...                             ....
            i=7, [7, 0]                     i=7, [7, 1]
*/
// clang-format on
std::tuple<uint32_t, uint32_t> get_coord_ref(int i, int wi_number) {
  return std::make_tuple(i, wi_number);
}

float sum_rows[MATRIX_M] = {0};

template <typename T1, typename T2, size_t M, size_t N, size_t K>
void matrix_multiply(big_matrix<T1, M, N> &C, big_matrix<T2, M, K> &A,
                     big_matrix<T2, K / 2, N * 2> &B) {
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  buffer<bfloat16, 2> bufA(A.get_data(), range<2>(M, K));
  buffer<bfloat16, 2> bufB(B.get_data(), range<2>(K, N));
  buffer<float, 2> bufC((float *)C.get_data(), range<2>(M, N));

  buffer<float> sum_rows_v(sum_rows, M); // there are total of M rows

  queue q;
  q.submit([&](handler &cgh) {
     auto accC = bufC.get_access<access::mode::read_write>(cgh);
     auto accA = bufA.get_access<access::mode::read_write>(cgh);
     auto accB = bufB.get_access<access::mode::read_write>(cgh);

     auto v = sum_rows_v.get_access<access::mode::read_write>(cgh);
     auto os = sycl::stream(100000, 6144, cgh);

     cgh.parallel_for<class imatrix>(
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
           joint_matrix<sub_group, bfloat16, use::a, TM, TK, layout::row_major>
               sub_a;
           // For B, we assume B has been already VNNIed.
           joint_matrix<sub_group, bfloat16, use::b, TK, TN,
                        ext::intel::experimental::matrix::layout::packed>
               sub_b;
           joint_matrix<sub_group, float, use::accumulator, TM, TN> sub_c;

           joint_matrix_load(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / SG_SZ * TN,
               N, layout::row_major);
           for (int k = 0; k < K / TK; k += 1) { //
             joint_matrix_load(
                 sg, sub_a,
                 accA.template get_multi_ptr<access::decorated::no>() +
                     (sg_startx * TM) * K + k * TK,
                 K);
             joint_matrix_load(
                 sg, sub_b,
                 accB.template get_multi_ptr<access::decorated::no>() +
                     (k * TK / 2) * (N * 2) + sg_starty / SG_SZ * TN * 2,
                 N * 2);
             sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
           }
           joint_matrix_store(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / SG_SZ * TN,
               N, layout::row_major);

           float sum_local_rows[M] = {0}; // 8 local rows, M total
           auto data =
               sycl::ext::intel::experimental::matrix::get_wi_data(sg, sub_c);

           // Keep track of rows handled in this WI
           int32_t handled_rows[M] = {-1};
           size_t
               global_index; // Index into the result array that holds the sums.

           for (int i = 0; i < data.length(); ++i) {
             auto dataItem = data[i];
             auto [row, col] = dataItem.get_coord();
             // get_coord_ref(i, spmd_item.get_local_id(1));
             global_index = row + global_idx * TM;

             sum_local_rows[global_index] += data[i];

             handled_rows[global_index] = 1;
           }

           for (int j = 0; j < M; j++) {
             if (handled_rows[j] == 1) {
               global_index = j;
               sum_local_rows[global_index] = reduce_over_group(
                   sg, sum_local_rows[global_index], sycl::plus<>());
               // only Groups leader perform the global reduction
               if (global_idy % SG_SZ == 0) {
                 sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device>
                     aref(v[global_index]);
                 aref.fetch_add(sum_local_rows[global_index]);
               }
             }
           }
         }); // parallel for
   }).wait();
}

bfloat16 A[MATRIX_M][MATRIX_K];
bfloat16 B[MATRIX_K / 2][MATRIX_N * 2];
float C[MATRIX_M][MATRIX_N];
float D[MATRIX_M][MATRIX_N];

float make_fp32(bfloat16 x) {
  unsigned int y = *((int *)&x);
  y = y << 16;
  float *res = reinterpret_cast<float *>(&y);
  return *res;
}

void matrix_multiply_ref(int *A_mem, int *B_mem, int *C_mem, int M, int N,
                         int K) {
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        // Because B was assumed VNNIed
        bfloat16 *va = (bfloat16 *)(A_mem + m * K + k);
        bfloat16 *vb = (bfloat16 *)(B_mem + k * N + n);
        float acc = *((float *)(C_mem + m * N + n));
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
      A[i][j] = bfloat16(1.0f * (i + j));
    }
  }
  for (int i = 0; i < MATRIX_K / 2; i++) {
    for (int j = 0; j < MATRIX_N * 2; j++) {
      B[i][j] = bfloat16(2.0f * i + 3.0f * j);
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
  matrix_multiply_ref((int32_t *)A, (int32_t *)B, (int32_t *)D, MATRIX_M,
                      MATRIX_N, MATRIX_K / 2);

  bool res = true;
  float sum_rows_ref[MATRIX_M] = {0};

  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      if (fabs(C[i][j] - D[i][j]) > BF16_EPSILON)
        res = false;
      sum_rows_ref[i] += C[i][j];
    }
    if (fabs(sum_rows_ref[i] - sum_rows[i]) > BF16_EPSILON)
      res = false;
  }
  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
}
