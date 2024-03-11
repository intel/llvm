using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

template <size_t TM, size_t TN, typename T1, size_t NUM_ROWS, size_t NUM_COLS>
void matrix_load_and_store(T1 *input, T1 *out_col_major, T1 *out_row_major,
                           queue q) {
  size_t M = NUM_ROWS;
  size_t N = NUM_COLS;

  static_assert((NUM_ROWS % TM) == 0);
  static_assert((NUM_COLS % TN) == 0);

  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;

  q.submit([&](handler &cgh) {
     cgh.parallel_for(
         nd_range<2>({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ}),
         [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           auto p_input =
               address_space_cast<sycl::access::address_space::global_space,
                                  sycl::access::decorated::no>(input);

           auto p_out_col_major =
               address_space_cast<sycl::access::address_space::global_space,
                                  sycl::access::decorated::no>(out_col_major);
           auto p_out_row_major =
               address_space_cast<sycl::access::address_space::global_space,
                                  sycl::access::decorated::no>(out_row_major);

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

           joint_matrix_load(sg, sub_matrix, p_input + col_major_offset, M,
                             layout::col_major);

           joint_matrix_store(sg, sub_matrix,
                              p_out_col_major + row_major_offset, N,
                              layout::row_major);

           joint_matrix_store(sg, sub_matrix,
                              p_out_row_major + col_major_offset, M,
                              layout::col_major);
         }); // parallel for
   }).wait();
}

template <size_t TM> void run_matrix_test() {
  static constexpr size_t MATRIX_M = TM * 16;
  static constexpr size_t MATRIX_N = TN * 16;

  queue q;
  float *input = malloc_shared<float>(MATRIX_M * MATRIX_N, q);
  float *out_col_major = malloc_shared<float>(MATRIX_M * MATRIX_N, q);
  float *out_row_major = malloc_shared<float>(MATRIX_M * MATRIX_N, q);
  float *ref_col_major = malloc_shared<float>(MATRIX_M * MATRIX_N, q);

  // input is column majot matrix so it is of NxM shape
  matrix_rand(MATRIX_N, MATRIX_M, input, (float)5.0);
  matrix_fill(MATRIX_M, MATRIX_N, out_col_major, (float)0);
  matrix_fill(MATRIX_N, MATRIX_M, out_row_major, (float)0);
  matrix_transpose(MATRIX_N, MATRIX_M, ref_col_major, input);

  matrix_load_and_store<TM, TN, float, MATRIX_M, MATRIX_N>(input, out_col_major,
                                                           out_row_major, q);

  // we use exact comparison as no low precision calculation is used in this
  // test
  std::cout << "compare results for TM " << TM << "\n";
  bool res = matrix_compare<float, float, true>(MATRIX_M, MATRIX_N,
                                                out_col_major, ref_col_major) &&
             matrix_compare<float, float, true>(MATRIX_N, MATRIX_M,
                                                out_row_major, input);
  free(input, q);
  free(out_col_major, q);
  free(out_row_major, q);
  free(ref_col_major, q);
  assert(res);
}

int main() {
  run_matrix_test<8>();
  run_matrix_test<7>();
  run_matrix_test<6>();
  run_matrix_test<5>();
  run_matrix_test<4>();
  run_matrix_test<3>();
  run_matrix_test<2>();
  run_matrix_test<1>();

  std::cout << "Passed\n";
  return 0;
}
