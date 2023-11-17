
template <typename T1, typename T2>
bool matrix_compare_exact(unsigned int rows, unsigned int cols, T1 *src,
                          T2 *ref) {
  bool res = true;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (src[i * cols + j] != ref[i * cols + j]) {
        res = false;
      }
    }
  }
  return res;
}

template <typename T>
void matrix_transpose(unsigned int rows, unsigned int cols, T *dst, T *src) {
  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      dst[i + j * rows] = src[i * cols + j];
    }
  }
}

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

template <size_t TM, size_t TN, typename T1, size_t NUM_ROWS, size_t NUM_COLS>
void matrix_load_and_store(T1 *input, T1 *out_col_major, T1 *out_row_major,
                           queue q) {
  size_t M = NUM_ROWS;
  size_t N = NUM_COLS;

  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;

  auto p_input = address_space_cast<sycl::access::address_space::global_space,
                                    sycl::access::decorated::no>(input);

  auto p_out_col_major =
      address_space_cast<sycl::access::address_space::global_space,
                         sycl::access::decorated::no>(out_col_major);
  auto p_out_row_major =
      address_space_cast<sycl::access::address_space::global_space,
                         sycl::access::decorated::no>(out_row_major);

  q.submit([&](handler &cgh) {
     cgh.parallel_for(
         nd_range<2>({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ}),
         [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, float, use::accumulator, TM, TN> sub_matrix;

           auto row_major_offset =
               (sg_startx * TM) * N + (sg_starty / SG_SZ * TN);
           auto col_major_offset =
               (sg_startx * TM) + (sg_starty / SG_SZ * TN) * M;

           joint_matrix_load(sg, sub_matrix, p_input + col_major_offset, N,
                             layout::col_major);

           joint_matrix_store(sg, sub_matrix,
                              p_out_col_major + row_major_offset, N,
                              layout::row_major);

           joint_matrix_store(sg, sub_matrix,
                              p_out_row_major + col_major_offset, N,
                              layout::col_major);
         }); // parallel for
   }).wait();
}

template <size_t TM> bool run_matrix_test() {
  static constexpr size_t MATRIX_M = 1024;
  static constexpr size_t MATRIX_N = 1024;

  queue q;
  float *input = malloc_shared<float>(MATRIX_M * MATRIX_N, q);
  float *out_col_major = malloc_shared<float>(MATRIX_M * MATRIX_N, q);
  float *out_row_major = malloc_shared<float>(MATRIX_M * MATRIX_N, q);
  float *ref_col_major = malloc_shared<float>(MATRIX_M * MATRIX_N, q);

  matrix_rand(MATRIX_M, MATRIX_N, input, (float)5.0);
  matrix_fill(MATRIX_M, MATRIX_N, out_col_major, (float)0);
  matrix_fill(MATRIX_M, MATRIX_N, out_row_major, (float)0);
  matrix_transpose(MATRIX_M, MATRIX_N, ref_col_major, input);

  matrix_load_and_store<TM, TN, float, MATRIX_M, MATRIX_N>(input, out_col_major,
                                                           out_row_major, q);

  // we use exact comparison as no low precision calculation is used in this
  // test
  bool res =
      matrix_compare_exact(MATRIX_M, MATRIX_N, out_col_major, ref_col_major) &&
      matrix_compare_exact(MATRIX_M, MATRIX_N, out_row_major, input);

  free(input, q);
  free(out_col_major, q);
  free(out_row_major, q);
  free(ref_col_major, q);
  return res;
}

int main() {
  bool res = true;
  if (res)
    res = run_matrix_test<8>();
  if (res)
    res = run_matrix_test<7>();
  if (res)
    res = run_matrix_test<6>();
  if (res)
    res = run_matrix_test<5>();
  if (res)
    res = run_matrix_test<4>();
  if (res)
    res = run_matrix_test<3>();
  if (res)
    res = run_matrix_test<2>();
  if (res)
    res = run_matrix_test<1>();

  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
}
