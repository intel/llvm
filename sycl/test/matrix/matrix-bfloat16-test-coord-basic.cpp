// RUN: %clangxx -fsycl -O2 -DSYCL_EXT_ONEAPI_MATRIX_VERSION=2 %s -o %t.out
// RUN: %t.out
// XFAIL: *

// this code calculates the sum of rows into a global array of number of rows
// elements. First, partial reduction is computed inside each SG, then atomic
// add is used to reduce between SG leaders.  The get_coord() API is used for
// retrieving the row

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

#define SG_SZ 16

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

template <typename T, size_t M, size_t N>
void sum_rows_ref(
    accessor<T, 2, access::mode::read, access::target::host_buffer> B,
    accessor<int, 1, access::mode::read, access::target::host_buffer>
        sum_rows) {
  int sum_rows_ref[M] = {0};
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      sum_rows_ref[i] += B[i][j];
    }
    auto diff = sum_rows[i] - sum_rows_ref[i];
    assert(std::fabs(static_cast<int>(diff)) <=
           std::numeric_limits<int>::epsilon());
  }
}

template <typename T, size_t M, size_t N>
void matrix_sum_rows(queue q, big_matrix<T, M, N> &B, nd_range<2> &r) {
  buffer<int8_t, 2> bufB(B.get_data(), range<2>(M, N));
  // size of vector is known because SG size of set by the user in this case
  int sum_rows[M] = {0};
  buffer<int> sum_rows_v(sum_rows, M); // there are total of tK/4 * 2, 16 rows
  q.submit([&](handler &cgh) {
     auto accB = bufB.get_access<access::mode::read_write>(cgh);

     auto v = sum_rows_v.get_access<access::mode::atomic>(cgh);

     cgh.parallel_for<class add_matrix>(
         r, [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           ext::oneapi::sub_group sg = spmd_item.get_sub_group();

           joint_matrix<T, TK, TN, use::b> sub_b(sg);

           joint_matrix_load(sg, sub_b,
                             accB.get_pointer() + (global_idx * (TK / 4) * N) +
                                 sg_starty / SG_SZ * TN * 4,
                             N, layout::packed_b);

           int32_t sum_local_rows[M] = {0};
           auto tBData = sub_b.get_wi_data();

           // each WI calculates local sum of rows
           for (int i = 0; i < tBData.length(); ++i) {
             // row and col holds global co_ordinates of the matrix
             auto [row, col] = tBData[i].get_coord();
             sum_local_rows[row] += tBData[i];

             sum_local_rows[row] =
                 reduce_over_group(sg, sum_local_rows[row], sycl::plus<>());
             // only Groups leader perform the global reduction
             if (global_idy % SG_SZ == 0) {
               atomic_fetch_add(v[row], sum_local_rows[row]);
             }
           }
         }); // parallel for
   }).wait();
  sum_rows_ref<T, M, N>(bufB.get_access<access::mode::read>(),
                        sum_rows_v.get_access<access::mode::read>());
}

static constexpr size_t MATRIX_K = TK / 4 * 2;
static constexpr size_t MATRIX_N = TN * 4 * 2;
int8_t B[MATRIX_K][MATRIX_N];

int main() {
  big_matrix<int8_t, MATRIX_K, MATRIX_N> MB((int8_t *)&B);

  size_t NDRangeK = MATRIX_K / (TK / 4);
  size_t NDRangeN = (MATRIX_N / 4) / TN;
  queue q;
  nd_range<2> r({NDRangeK, NDRangeN * SG_SZ}, {1, 1 * SG_SZ});

  for (int i = 0; i < MATRIX_K; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      B[i][j] = i;
    }
  }

  matrix_sum_rows<int8_t, MATRIX_K, MATRIX_N>(q, MB, r);

  return 0;
}