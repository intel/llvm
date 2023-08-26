#define TM 8
#define TK 32

template <typename T, size_t NUM_ROWS, size_t NUM_COLS> struct big_matrix {
public:
  T *mat;

public:
  T *get_data() { return mat; }
  void set_data(T *data) { mat = data; }
  big_matrix(T *data) : mat(data) {}
};

template <typename T, size_t M, size_t N>
void assert_ops_ref(host_accessor<T, 2, access::mode::read> C, const int ref) {
  for (size_t i = 0; i < M; i++)
    for (size_t j = 0; j < N; j++) {
      auto diff = C[i][j] - ref;
      assert(std::fabs(static_cast<int>(diff)) <=
             std::numeric_limits<int>::epsilon());
    }
}
template <typename T, size_t M, size_t N>
void matrix_verify_add(queue q, big_matrix<T, M, N> &A, nd_range<2> &r,
                       const int ref) {
  buffer<int8_t, 2> bufB(A.get_data(), range<2>(M, N));

  q.submit([&](handler &cgh) {
     auto accA = bufB.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for<class add_matrix>(
         r, [accA](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, int8_t, use::b, TK, TN,
                        ext::intel::experimental::matrix::layout::packed>
               sub_b;

           joint_matrix_fill(sg, sub_b, 5);

           auto wi_slice_b =
               sycl::ext::intel::experimental::matrix::get_wi_data(sg, sub_b);
           for (int i = 0; i < wi_slice_b.length(); i++) {
             wi_slice_b[i] = wi_slice_b[i] + 2;
           }
           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_b,
               accA.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N * 4 + sg_starty / SG_SZ * TN * 4,
               N * 4);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, M, N>(bufB.get_host_access(read_only), ref);
}

template <typename T, size_t M, size_t N>
void matrix_verify_sub(queue q, big_matrix<T, M, N> &A, nd_range<2> &r,
                       const int ref) {
  buffer<int8_t, 2> bufB(A.get_data(), range<2>(M, N));

  q.submit([&](handler &cgh) {
     auto accA = bufB.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for<class sub_matrix>(
         r, [accA](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, int8_t, use::b, TK, TN,
                        ext::intel::experimental::matrix::layout::packed>
               sub_b;

           joint_matrix_fill(sg, sub_b, 5);

           auto wi_slice_b =
               sycl::ext::intel::experimental::matrix::get_wi_data(sg, sub_b);
           for (int i = 0; i < wi_slice_b.length(); i++) {
             wi_slice_b[i] = wi_slice_b[i] - 2;
           }
           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_b,
               accA.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N * 4 + sg_starty / SG_SZ * TN * 4,
               N * 4);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, M, N>(bufB.get_host_access(read_only), ref);
}

template <typename T, size_t M, size_t N>
void matrix_verify_mul(queue q, big_matrix<T, M, N> &A, nd_range<2> &r,
                       const int ref) {
  buffer<int8_t, 2> bufB(A.get_data(), range<2>(M, N));

  q.submit([&](handler &cgh) {
     auto accA = bufB.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for<class mul_matrix>(
         r, [accA](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, int8_t, use::b, TK, TN,
                        ext::intel::experimental::matrix::layout::packed>
               sub_b;

           joint_matrix_fill(sg, sub_b, 5);

           auto wi_slice_b =
               sycl::ext::intel::experimental::matrix::get_wi_data(sg, sub_b);
           for (int i = 0; i < wi_slice_b.length(); i++) {
             wi_slice_b[i] = wi_slice_b[i] * 3;
           }
           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_b,
               accA.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N * 4 + sg_starty / SG_SZ * TN * 4,
               N * 4);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, M, N>(bufB.get_host_access(read_only), ref);
}

template <typename T, size_t M, size_t N>
void matrix_verify_div(queue q, big_matrix<T, M, N> &A, nd_range<2> &r,
                       const int ref) {
  buffer<int8_t, 2> bufB(A.get_data(), range<2>(M, N));

  q.submit([&](handler &cgh) {
     auto accA = bufB.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for<class div_matrix>(
         r, [accA](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, int8_t, use::b, TK, TN,
                        ext::intel::experimental::matrix::layout::packed>
               sub_b;

           joint_matrix_fill(sg, sub_b, 4);

           auto wi_slice_b =
               sycl::ext::intel::experimental::matrix::get_wi_data(sg, sub_b);
           for (int i = 0; i < wi_slice_b.length(); i++) {
             wi_slice_b[i] = wi_slice_b[i] / 2;
           }
           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_b,
               accA.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N * 4 + sg_starty / SG_SZ * TN * 4,
               N * 4);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, M, N>(bufB.get_host_access(read_only), ref);
}

template <typename T, size_t M, size_t N>
void matrix_verify_logic(queue q, big_matrix<T, M, N> &A, nd_range<2> &r,
                         const int ref) {
  buffer<int8_t, 2> bufB(A.get_data(), range<2>(M, N));

  q.submit([&](handler &cgh) {
     auto accA = bufB.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for<class logic_matrix>(
         r, [accA](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, int8_t, use::b, TK, TN,
                        ext::intel::experimental::matrix::layout::packed>
               sub_b;

           joint_matrix_fill(sg, sub_b, 5);

           auto wi_slice_b =
               sycl::ext::intel::experimental::matrix::get_wi_data(sg, sub_b);
           for (int i = 0; i < wi_slice_b.length(); i++) {
             if (wi_slice_b[i]) {
               if (wi_slice_b[i] > 2 || wi_slice_b[i] >= 2 ||
                   wi_slice_b[i] < 2 || wi_slice_b[i] <= 2) {
                 T val = (wi_slice_b[i] != 2) ? wi_slice_b[i] : 2;
                 val--;
                 val++;
                 if (wi_slice_b[i] == 2) {
                   val -= 2;
                   val *= 3;
                   val /= 2;
                 } else {
                   val += 2;
                 }
                 wi_slice_b[i] = val;
               }
             }
           }
           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_b,
               accA.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N * 4 + sg_starty / SG_SZ * TN * 4,
               N * 4);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, M, N>(bufB.get_host_access(read_only), ref);
}

static constexpr size_t MATRIX_M = TM * 2;
static constexpr size_t MATRIX_N = TN * 2;
int8_t B[MATRIX_M][MATRIX_N];
int D[MATRIX_M][MATRIX_N];

int main() {

  big_matrix<int, MATRIX_M, MATRIX_N> MD((int *)&D);
  big_matrix<int8_t, MATRIX_M, MATRIX_N> MB((int8_t *)&B);

  size_t NDRangeM = MATRIX_M / TM;
  size_t NDRangeN = MATRIX_N / TN;
  queue q;
  nd_range<2> r({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ});

  matrix_verify_add<int8_t, MATRIX_M, MATRIX_N>(q, MB, r, 7);
  matrix_verify_sub<int8_t, MATRIX_M, MATRIX_N>(q, MB, r, 3);
  matrix_verify_mul<int8_t, MATRIX_M, MATRIX_N>(q, MB, r, 15);
  matrix_verify_div<int8_t, MATRIX_M, MATRIX_N>(q, MB, r, 2);
  matrix_verify_logic<int8_t, MATRIX_M, MATRIX_N>(q, MB, r, 7);

  return 0;
}
