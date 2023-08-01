#define TM 8
#define TN SG_SZ
#define TK 16

static float make_fp32(bfloat16 x) {
  unsigned int y = *((int *)&x);
  y = y << 16;
  float *res = reinterpret_cast<float *>(&y);
  return *res;
}

template <typename T, size_t NUM_ROWS, size_t NUM_COLS> struct big_matrix {
public:
  T *mat;

public:
  T *get_data() { return mat; }
  void set_data(T *data) { mat = data; }
  big_matrix(T *data) : mat(data) {}
};

template <typename T, size_t NUM_ROWS, size_t NUM_COLS>
void assert_ops_ref(host_accessor<T, 2, access::mode::read> mat,
                    const float ref) {
  for (size_t i = 0; i < NUM_ROWS; i++)
    for (size_t j = 0; j < NUM_COLS; j++) {
      float diff;
      if constexpr (std::is_same_v<T, bfloat16>)
        diff = make_fp32(mat[i][j]) - ref;
      else
        diff = mat[i][j] - ref;
      assert(std::fabs(static_cast<float>(diff)) <
             std::numeric_limits<float>::epsilon());
    }
}

template <typename T, size_t NUM_ROWS, size_t NUM_COLS, size_t SUB_ROWS, size_t SUB_COLS, typename joint_matrix_t, use Use, typename OP>
void matrix_verify_op(queue q, big_matrix<T, NUM_ROWS, NUM_COLS> &mat, nd_range<2> &r,
                       const float ref, OP op) {
  buffer<T, 2> bufMat(mat.get_data(), range<2>(NUM_ROWS, NUM_COLS));

  q.submit([&](handler &cgh) {
     sycl::accessor accessMat{bufMat, cgh, sycl::read_write};
     cgh.parallel_for(
         r, [accessMat, op](nd_item<2> spmd_item)[[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix_t sub_mat;
           if constexpr (std::is_same_v<T, bfloat16>)
             joint_matrix_fill(sg, sub_mat, bfloat16(5.0));
           else
             joint_matrix_fill(sg, sub_mat, 5);
           auto wi_slice =
               sycl::ext::intel::experimental::matrix::get_wi_data(sg, sub_mat);
           for (int i = 0; i < wi_slice.length(); i++) {
             if constexpr (std::is_same_v<T, bfloat16>)
               wi_slice[i] = op(wi_slice[i], bfloat16(2));
             else
               wi_slice[i] = op(wi_slice[i], 2);
           }

           //if (Use == use::a) {

            ext::intel::experimental::matrix::joint_matrix_store(
                sg, sub_mat,
                accessMat.template get_multi_ptr<access::decorated::no>() +
                    (sg_startx * SUB_ROWS) * NUM_COLS + sg_starty / SG_SZ * SUB_COLS,
                NUM_COLS);
          //  } else {
          //   joint_matrix_store(
          //       sg, sub_mat,
          //       accessMat.template get_multi_ptr<access::decorated::no>() +
          //           (sg_startx * SUB_ROWS) * NUM_COLS + sg_starty / SG_SZ * SUB_COLS,
          //       NUM_COLS, layout::row_major);

          //  }

         }); // parallel for
   })
      .wait();
  assert_ops_ref<T, NUM_ROWS, NUM_COLS>(bufMat.get_host_access(read_only), ref);
}
/*
template <typename T, size_t NUM_ROWS, size_t NUM_COLS, size_t SUB_ROWS, size_t SUB_COLS, typename joint_matrix_t, use Use>
void matrix_verify_logic(queue q, big_matrix<T, NUM_ROWS, NUM_COLS> &mat, nd_range<2> &r,
                         const float ref) {
  buffer<T, 2> bufMat(mat.get_data(), range<2>(NUM_ROWS, NUM_COLS));

  q.submit([&](handler &cgh) {
     sycl::accessor accessMat{bufMat, cgh, sycl::read_write};
     cgh.parallel_for(
         r, [accessMat](nd_item<2> spmd_item)[[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix_t sub_mat;
           if constexpr (std::is_same_v<T, bfloat16>)
             joint_matrix_fill(sg, sub_mat, bfloat16(5.0));
           else
             joint_matrix_fill(sg, sub_mat, 5);
           auto wi_slice =
               sycl::ext::intel::experimental::matrix::get_wi_data(sg, sub_mat);
           for (int i = 0; i < wi_slice.length(); i++) {
             if (wi_slice[i]) {
               if constexpr (std::is_same_v<T, bfloat16>) {
                 if (wi_slice[i] > bfloat16(2.0) ||
                     wi_slice[i] >= bfloat16(2.0) ||
                     wi_slice[i] < bfloat16(2.0) ||
                     wi_slice[i] <= bfloat16(2.0)) {
                   T val = (wi_slice[i] != bfloat16(2.0)) ? wi_slice[i]
                                                            : bfloat16(2.0);
                   val = bfloat16(make_fp32(val) - static_cast<float>(1));
                   val = bfloat16(make_fp32(val) + static_cast<float>(1));
                   if (wi_slice[i] == bfloat16(2.0)) {
                     val = bfloat16(make_fp32(val) - static_cast<float>(2));
                     val = bfloat16(make_fp32(val) * static_cast<float>(3));
                     val = bfloat16(make_fp32(val) / static_cast<float>(2));

                   } else {
                     val = bfloat16(make_fp32(val) + static_cast<float>(2));
                   }
                   wi_slice[i] = val;
                 }
               } else {
                 if (wi_slice[i] > 2.0 || wi_slice[i] >= 2.0 ||
                     wi_slice[i] < 2.0 || wi_slice[i] <= 2.0) {
                   T val = (wi_slice[i] != 2.0) ? wi_slice[i]
                                                  : static_cast<T>(2.0);
                   val = val - 1;
                   val = val + 1;
                   if (wi_slice[i] == 2.0) {
                     val = val - 2;
                     val = val * 3;
                     val = val / 2;
                   } else {
                     val = val + 2;
                   }
                   wi_slice[i] = val;
                 }
               }
             }
           }
           if (Use == use::a) {
            ext::intel::experimental::matrix::joint_matrix_store(
                sg, sub_mat,
                accessMat.template get_multi_ptr<access::decorated::no>() +
                    (sg_startx * SUB_ROWS) * NUM_COLS + sg_starty / SG_SZ * SUB_COLS,
                NUM_COLS);
           } else {
            joint_matrix_store(
                sg, sub_mat,
                accessMat.template get_multi_ptr<access::decorated::no>() +
                    (sg_startx * SUB_ROWS) * NUM_COLS + sg_starty / SG_SZ * SUB_COLS,
                NUM_COLS, layout::row_major);

           }
         }); // parallel for
   })
      .wait();
  assert_ops_ref<T, NUM_ROWS, NUM_COLS>(bufMat.get_host_access(read_only), ref);
}
*/
template <typename T, size_t NUM_ROWS, size_t NUM_COLS, size_t SUB_ROWS, size_t SUB_COLS, typename joint_matrix_t, use Use> void test_ewops() {
  T mat[NUM_ROWS][NUM_COLS];
  big_matrix<T, NUM_ROWS, NUM_COLS> big_mat((T *)&mat);

  size_t NDRangeRows = NUM_ROWS / SUB_ROWS;
  size_t NDRangeCols = NUM_COLS / SUB_COLS;

  queue q;
  nd_range<2> r({NDRangeRows, NDRangeCols * SG_SZ}, {1, 1 * SG_SZ});

  std::cout << "+: ";
  matrix_verify_op<T, NUM_ROWS, NUM_COLS, SUB_ROWS, SUB_COLS, joint_matrix_t, Use>(q, big_mat, r, 7.0, [](auto l, auto r){ return l + r;});
  std::cout << "passed\n";
  std::cout << "-: ";
  matrix_verify_op<T, NUM_ROWS, NUM_COLS, SUB_ROWS, SUB_COLS, joint_matrix_t, Use>(q, big_mat, r, 3.0, [](auto l, auto r){ return l - r;});
  std::cout << "passed\n";
  std::cout << "*: ";
  matrix_verify_op<T, NUM_ROWS, NUM_COLS, SUB_ROWS, SUB_COLS, joint_matrix_t, Use>(q, big_mat, r, 10.0, [](auto l, auto r){ return l * r;});
  std::cout << "passed\n";
  std::cout << "/: ";
  matrix_verify_op<T, NUM_ROWS, NUM_COLS, SUB_ROWS, SUB_COLS, joint_matrix_t, Use>(q, big_mat, r, 2.5, [](auto l, auto r){ return l / r;});
  std::cout << "passed\n";
  //matrix_verify_logic<T, NUM_ROWS, NUM_COLS, SUB_ROWS, SUB_COLS, joint_matrix_t, Use>(q, big_mat, r, 7.0);
}

int main() {
  static constexpr size_t MATRIX_M = TM * 2;
  static constexpr size_t MATRIX_N = TN * 2;
  static constexpr size_t MATRIX_K = TK * 2;

  // test A
  test_ewops<bfloat16, MATRIX_M, MATRIX_K, TM, TK, joint_matrix<sub_group, bfloat16, use::a, TM, TK, layout::row_major>, use::a>();

  // test C
  //test_ewops<bfloat16, MATRIX_M, MATRIX_N, TM, TN, joint_matrix<sub_group, bfloat16, use::accumulator, TM, TN>, use::accumulator>();
  //test_ewops<float, MATRIX_M, MATRIX_N, TM, TN, joint_matrix<sub_group, float, use::accumulator, TM, TN>, use::accumulator>();

  return 0;
}
