
#define TM 8
#define TN SG_SZ

template <typename T, size_t NUM_ROWS, size_t NUM_COLS> struct big_matrix {
public:
  T *mat;

public:
  T *get_data() { return mat; }
  void set_data(T *data) { mat = data; }
  big_matrix(T *data) : mat(data) {}
};

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
template <typename T, size_t M, size_t N>
void matrix_verify_add(queue q, big_matrix<T, M, N> &A, nd_range<2> &r,
                       const float ref) {
  buffer<float, 2> bufC(A.get_data(), range<2>(M, N));

  q.submit([&](handler &cgh) {
     auto accC = bufC.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for<class add_matrix>(
         r, [accC](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T, use::accumulator, TM, TN> sub_c;

           joint_matrix_fill(sg, sub_c, 5.0);

           auto wi_slice_c =
               sycl::ext::intel::experimental::matrix::get_wi_data(sg, sub_c);
           for (int i = 0; i < wi_slice_c.length(); i++) {
             wi_slice_c[i] = wi_slice_c[i] + 2.0;
           }

           joint_matrix_store(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / SG_SZ * TN,
               N, layout::row_major);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, M, N>(bufC.get_host_access(read_only), ref);
}

template <typename T, size_t M, size_t N>
void matrix_verify_sub(queue q, big_matrix<T, M, N> &C, nd_range<2> &r,
                       const float ref) {
  buffer<float, 2> bufC(C.get_data(), range<2>(M, N));

  q.submit([&](handler &cgh) {
     auto accC = bufC.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for<class sub_matrix>(
         r, [accC](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T, use::accumulator, TM, TN> sub_c;

           joint_matrix_fill(sg, sub_c, 5.0);

           auto wi_slice_c =
               sycl::ext::intel::experimental::matrix::get_wi_data(sg, sub_c);
           for (int i = 0; i < wi_slice_c.length(); i++) {
             wi_slice_c[i] = wi_slice_c[i] - 2;
           }
           joint_matrix_store(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / SG_SZ * TN,
               N, layout::row_major);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, M, N>(bufC.get_host_access(read_only), ref);
}

template <typename T, size_t M, size_t N>
void matrix_verify_mul(queue q, big_matrix<T, M, N> &C, nd_range<2> &r,
                       const float ref) {
  buffer<float, 2> bufC(C.get_data(), range<2>(M, N));

  q.submit([&](handler &cgh) {
     auto accC = bufC.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for<class mul_matrix>(
         r, [accC](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T, use::accumulator, TM, TN> sub_c;
           joint_matrix_fill(sg, sub_c, float(5.0));

           auto wi_slice_c =
               sycl::ext::intel::experimental::matrix::get_wi_data(sg, sub_c);
           for (int i = 0; i < wi_slice_c.length(); i++) {
             wi_slice_c[i] = wi_slice_c[i] * float(3.0);
           }
           joint_matrix_store(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / SG_SZ * TN,
               N, layout::row_major);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, M, N>(bufC.get_host_access(read_only), ref);
}

template <typename T, size_t M, size_t N>
void matrix_verify_div(queue q, big_matrix<T, M, N> &C, nd_range<2> &r,
                       const float ref) {
  buffer<float, 2> bufC(C.get_data(), range<2>(M, N));

  q.submit([&](handler &cgh) {
     auto accC = bufC.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for<class div_matrix>(
         r, [accC](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T, use::accumulator, TM, TN> sub_c;

           joint_matrix_fill(sg, sub_c, float(4.0));

           auto wi_slice_c =
               sycl::ext::intel::experimental::matrix::get_wi_data(sg, sub_c);
           for (int i = 0; i < wi_slice_c.length(); i++) {
             wi_slice_c[i] = wi_slice_c[i] / float(2.0);
           }
           joint_matrix_store(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / SG_SZ * TN,
               N, layout::row_major);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, M, N>(bufC.get_host_access(read_only), ref);
}

template <typename T, size_t M, size_t N>
void matrix_verify_logic(queue q, big_matrix<T, M, N> &C, nd_range<2> &r,
                         const float ref) {
  buffer<float, 2> bufC(C.get_data(), range<2>(M, N));

  q.submit([&](handler &cgh) {
     auto accC = bufC.get_access<access::mode::read_write>(cgh);
     cgh.parallel_for<class logic_matrix>(
         r, [accC](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T, use::accumulator, TM, TN> sub_c;

           joint_matrix_fill(sg, sub_c, float(5.0));

           auto wi_slice_c =
               sycl::ext::intel::experimental::matrix::get_wi_data(sg, sub_c);
           for (int i = 0; i < wi_slice_c.length(); i++) {
             if (wi_slice_c[i]) {
               if (wi_slice_c[i] > float(2.0) || wi_slice_c[i] >= float(2.0) ||
                   wi_slice_c[i] < float(2.0) || wi_slice_c[i] <= float(2.0)) {
                 T val =
                     (wi_slice_c[i] != float(2.0)) ? wi_slice_c[i] : float(2.0);
                 val = val - static_cast<float>(1);
                 val = val + static_cast<float>(1);
                 if (wi_slice_c[i] == float(2.0)) {
                   val = val - static_cast<float>(2);
                   val = val * static_cast<float>(3);
                   val = val / static_cast<float>(2);

                 } else {
                   val = val + static_cast<float>(2);
                 }
                 wi_slice_c[i] = val;
               }
             }
           }
           joint_matrix_store(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / SG_SZ * TN,
               N, layout::row_major);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, M, N>(bufC.get_host_access(read_only), ref);
}

static constexpr size_t MATRIX_M = TM * 2;
static constexpr size_t MATRIX_N = TN * 2;
float C[MATRIX_M][MATRIX_N];
float D[MATRIX_M][MATRIX_N];

void matrix_ops_ref(float *D, int M, int N) {
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      *(D + m * N + n) = 0;
      *(D + m * N + n) *= 2;
    }
}

int main() {

  big_matrix<float, MATRIX_M, MATRIX_N> MD((float *)&D);
  big_matrix<float, MATRIX_M, MATRIX_N> MC((float *)&C);

  size_t NDRangeM = MATRIX_M / TM;
  size_t NDRangeN = MATRIX_N / TN;
  queue q;
  nd_range<2> r({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ});

  matrix_verify_add<float, MATRIX_M, MATRIX_N>(q, MC, r, 7.0);
  matrix_verify_sub<float, MATRIX_M, MATRIX_N>(q, MC, r, 3.0);
  matrix_verify_mul<float, MATRIX_M, MATRIX_N>(q, MC, r, 15.0);
  matrix_verify_div<float, MATRIX_M, MATRIX_N>(q, MC, r, 2.0);
  matrix_verify_logic<float, MATRIX_M, MATRIX_N>(q, MC, r, 7.0);

  return 0;
}
