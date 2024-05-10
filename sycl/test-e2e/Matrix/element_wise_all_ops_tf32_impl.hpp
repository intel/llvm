#define TM 8
#define TK 8

template <typename T, size_t M, size_t N>
void assert_ops_ref(host_accessor<T, 2, access::mode::read> C,
                    const float ref) {
  for (size_t i = 0; i < M; i++)
    for (size_t j = 0; j < N; j++) {
      auto diff = C[i][j] - ref;
      assert(std::fabs(static_cast<float>(diff)) <
             std::numeric_limits<float>::epsilon());
    }
}

template <typename T, typename Ts, size_t M, size_t K>
void matrix_verify_add(queue q, big_matrix<Ts, M, K> &A, nd_range<2> &r,
                       const float ref) {
  buffer<Ts, 2> bufA(A.get_data(), range<2>(M, K));

  q.submit([&](handler &cgh) {
     sycl::accessor accA{bufA, cgh, sycl::read_write};

     cgh.parallel_for<class add_matrix>(
         r, [accA](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T, use::a, TM, TK, layout::row_major> sub_a;

           joint_matrix_fill(sg, sub_a, round_to_tf32(5.0));

           joint_matrix_apply(sg, sub_a,
                              [&](float &x) { x = x + round_to_tf32(2); });

           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_a,
               accA.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * K + sg_starty / SG_SZ * TK,
               K);
         }); // parallel for
   }).wait();
  assert_ops_ref<Ts, M, K>(bufA.get_host_access(sycl::read_only), ref);
}

template <typename T, typename Ts, size_t M, size_t K>
void matrix_verify_sub(queue q, big_matrix<Ts, M, K> &A, nd_range<2> &r,
                       const float ref) {
  buffer<Ts, 2> bufA(A.get_data(), range<2>(M, K));

  q.submit([&](handler &cgh) {
     sycl::accessor accA{bufA, cgh, sycl::read_write};

     cgh.parallel_for<class sub_matrix>(
         r, [accA](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T, use::a, TM, TK, layout::row_major> sub_a;

           joint_matrix_fill(sg, sub_a, round_to_tf32(5.0));

           joint_matrix_apply(sg, sub_a,
                              [&](float &x) { x = x - round_to_tf32(2); });

           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_a,
               accA.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * K + sg_starty / SG_SZ * TK,
               K);
         }); // parallel for
   }).wait();
  assert_ops_ref<Ts, M, K>(bufA.get_host_access(sycl::read_only), ref);
}

template <typename T, typename Ts, size_t M, size_t K>
void matrix_verify_mul(queue q, big_matrix<Ts, M, K> &A, nd_range<2> &r,
                       const float ref) {
  buffer<Ts, 2> bufA(A.get_data(), range<2>(M, K));

  q.submit([&](handler &cgh) {
     sycl::accessor accA{bufA, cgh, sycl::read_write};

     cgh.parallel_for<class mul_matrix>(
         r, [accA](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T, use::a, TM, TK, layout::row_major> sub_a;
           joint_matrix_fill(sg, sub_a, round_to_tf32(5.0));

           joint_matrix_apply(sg, sub_a,
                              [&](float &x) { x = x * round_to_tf32(3.0); });
           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_a,
               accA.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * K + sg_starty / SG_SZ * TK,
               K);
         }); // parallel for
   }).wait();
  assert_ops_ref<Ts, M, K>(bufA.get_host_access(sycl::read_only), ref);
}

template <typename T, typename Ts, size_t M, size_t K>
void matrix_verify_div(queue q, big_matrix<Ts, M, K> &A, nd_range<2> &r,
                       const float ref) {
  buffer<Ts, 2> bufA(A.get_data(), range<2>(M, K));

  q.submit([&](handler &cgh) {
     sycl::accessor accA{bufA, cgh, sycl::read_write};

     cgh.parallel_for<class div_matrix>(
         r, [accA](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T, use::a, TM, TK, layout::row_major> sub_a;

           joint_matrix_fill(sg, sub_a, round_to_tf32(4.0));

           joint_matrix_apply(sg, sub_a,
                              [&](float &x) { x = x / round_to_tf32(2); });
           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_a,
               accA.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * K + sg_starty / SG_SZ * TK,
               K);
         }); // parallel for
   }).wait();
  assert_ops_ref<Ts, M, K>(bufA.get_host_access(sycl::read_only), ref);
}

template <typename T, typename Ts, size_t M, size_t K>
void matrix_verify_logic(queue q, big_matrix<Ts, M, K> &A, nd_range<2> &r,
                         const float ref) {
  buffer<Ts, 2> bufA(A.get_data(), range<2>(M, K));

  q.submit([&](handler &cgh) {
     sycl::accessor accA{bufA, cgh, sycl::read_write};
     cgh.parallel_for<class logic_matrix>(
         r, [accA](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T, use::a, TM, TK, layout::row_major> sub_a;

           joint_matrix_fill(sg, sub_a, round_to_tf32(5.0));

           joint_matrix_apply(sg, sub_a, [&](float &x) {
             if (x) {
               if (x > 2 || x >= 2 || x < 2 || x <= 2) {
                 float val = (x != 2) ? x : 2;
                 val--;
                 val++;
                 if (x == 2) {
                   val -= 2;
                   val *= 3;
                   val /= 2;
                 } else {
                   val += 2;
                 }
                 x = val;
               }
             }
           });
           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_a,
               accA.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * K + sg_starty / SG_SZ * TK,
               K);
         }); // parallel for
   }).wait();
  assert_ops_ref<Ts, M, K>(bufA.get_host_access(sycl::read_only), ref);
}

template <typename T, typename Ts, size_t M, size_t K, size_t TileM,
          size_t TileN, size_t TileK, class kernel_name, typename OP>
void matrix_verify_op(big_matrix<Ts, M, K> &A, const float ref, OP op) {
  buffer<Ts, 2> bufA(A.get_data(), range<2>(M, K));

  queue q;
  size_t sg_size = get_sg_size<kernel_name>(q);
  nd_range<2> r({M / TileM, K / TileK * sg_size}, {1, 1 * sg_size});

  q.submit([&](handler &cgh) {
     sycl::accessor accA{bufA, cgh, sycl::read_write};

     cgh.parallel_for<class mul_matrix>(
         r, [accA](nd_item<2> spmd_item)
#ifdef SG_SZ
                [[intel::reqd_sub_group_size(SG_SZ)]]
#endif
         {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T, use::a, TileM, TileK, layout::row_major>
               sub_a;
           joint_matrix_fill(sg, sub_a, round_to_tf32(5.0));

           joint_matrix_apply(sg, sub_a, op);
           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_a,
               accA.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TileM) * K + sg_starty / sg_size * TileK,
               K);
         }); // parallel for
   }).wait();
  assert_ops_ref<Ts, M, K>(bufA.get_host_access(sycl::read_only), ref);
}

static constexpr size_t MATRIX_M = TM * 2;
static constexpr size_t MATRIX_K = TK * 2;
float A[MATRIX_M][MATRIX_K];
float D[MATRIX_M][MATRIX_K];

int main() {

  big_matrix<float, MATRIX_M, MATRIX_K> MD((float *)&D);
  big_matrix<float, MATRIX_M, MATRIX_K> MA((float *)&A);

  // size_t NDRangeM = MATRIX_M / TM;
  // size_t NDRangeK = MATRIX_K / TK;
  // queue q;
  // nd_range<2> r({NDRangeM, NDRangeK * SG_SZ}, {1, 1 * SG_SZ});

  // matrix_verify_add<precision::tf32, float, MATRIX_M, MATRIX_K>(q, MA,
  // r, 7.0);
  matrix_verify_op<precision::tf32, float, MATRIX_M, MATRIX_K, TM, TN, TK,
                   class add>(MA, 7.0,
                              [=](auto &x) { x = x + round_to_tf32(2); });

  // matrix_verify_sub<precision::tf32, float, MATRIX_M, MATRIX_K>(q, MA,
  // r, 3.0); matrix_verify_mul<precision::tf32, float, MATRIX_M, MATRIX_K>(q,
  // MA, r, 15.0); matrix_verify_div<precision::tf32, float, MATRIX_M,
  // MATRIX_K>(q, MA, r, 2.0); matrix_verify_logic<precision::tf32, float,
  // MATRIX_M, MATRIX_K>(q, MA, r,
  //                                                                 7.0);

  return 0;
}
