// RUN: %clangxx -fsycl -O2 %s -o %t.out

// Kernel B sum by col
#include <cmath>
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
void sum_cols_ref(host_accessor<T, 2, access::mode::read_write> B,
                  host_accessor<int, 1, access::mode::read_write> sum_cols) {
  int sum_cols_ref[N] = {0};
  for (size_t j = 0; j < N; j++) {
    for (size_t i = 0; i < M; i++) {
      sum_cols_ref[j] += B[i][j];
    }
    auto diff = sum_cols[j] - sum_cols_ref[j];
    assert(std::fabs(static_cast<int>(diff)) <=
           std::numeric_limits<int>::epsilon());
  }
}

// clang-format off
/* 
    Here is a demonstration of how matrix B will be divided across
    work items for this test case.
    <    ---------------    128    ---------------------------------->
    x x x x x x x x x x x x x x x x       ..........    x x x x x x   ^
    x x x x x x x x x x x x x x x x       ..........    x x x x x x  16
    x x x x x x x x x x x x x x x x       ..........    x x x x x x   |
    .....                                                             |
    x x x x x x x x x x x x x x x x       ..........    x x x x x x   |
    x x x x x x x x x x x x x x x x       ..........    x x x x x x   v

    
    ---------------    64    ---------------->
    x x x x   x x    ..........    x x  x x x x   ^
    x x x x   x x    ..........    x x  x x x x   8
    x x x x   x x    ..........    x x  x x x x   |  <-- part of (VNNI-ed) 
    .....                                         |   original matrix each SG
    x x x x   x x    ..........    x x  x x x x   |   holds
    x x x x   x x    ..........    x x  x x x x   v
    < WI0 >                            < WI15 >


    <--------    16    ------------->
    x x x     ..........    x x x   ^
    x x x     ..........    x x x   |
    x x x     ..........    x x x   | <-- part of (non-VNNI-ed) original matrix
    .....                           |           each SG holds
    x x x     ..........    x x x   |
    x x x     ..........    x x x   |
    x x x     ..........    x x x  32
    x x x     ..........    x x x   |
    x x x     ..........    x x x   |
    x x x     ..........    x x x   |
    x x x     ..........    x x x   |
    x x x     ..........    x x x   |
    x x x     ..........    x x x   v

    If we dividie the above matrix across 16 (SG_SZ) work items,
    each WI will hold 32 elements.  And these 32 elements will be
    8x4 chunks as shown in the VNNI-ed matrix figure. 
*/

// The total distribution among the WIs in ALL the sub-groups is as follows:
// This is useful to figure out the the global index is to be calculated

/*
W0 --> 0 0 0 0   1 1 1 1 ...   7 7 7 7 --> total 32 elements
wi [0,0] --> i=0, [0, 0]        wi [0,1] --> i=0, [0, 4]     wi [0,15] --> i=0, [0, 60] | wi [0,16] --> i=0, [0, 64]
            i=1, [0, 1]                     i=1, [0, 5]                   i=1, [0, 61]  |               i=1, [0, 65]
            i=2, [0, 2]                     i=2, [0, 6]                   i=2, [0, 62]  |               i=2, [0, 66]
            i=3, [0, 3]                     i=3, [0, 7]                   i=3, [0, 63]  |               i=3, [0, 67]              

            i=4, [1, 0]                     i=4, [1, 4]                   i=4, [1, 60]  |               ....
            i=5, [1, 1]                     i=5, [1, 5]                   i=5, [1, 61]  |
            i=6, [1, 2]                     i=6, [1, 6]                   i=6, [1, 62]  |
            i=7, [1, 3]                     i=7, [1, 7]                   i=7, [1, 63]  |
            ...                             ...                           ....          |
            i=28,[7, 0]                     i=28,[7, 4]                   i=28,[7, 60]  |               i=28, [7, 124]
            i=29,[7, 1]                     i=29,[7, 5]                   i=29,[7, 61]  |               i=29, [7, 125]
            i=30,[7, 2]                     i=30,[7, 6]                   i=30,[7, 62]  |               i=30, [7, 126]
            i=31,[7, 3]                     i=31,[7, 7]                   i=31,[7, 63]  |               i=31, [7, 127]
---------------------------------------------------------------------------------------- ---------------------------
wi [1,0] -->    i=0, [8, 0]
                i=1, [8, 1]
                i=2, [8, 2]
                i=3, [8, 2]
                ...
                i=28, [15, 0]
                i=29, [15, 1]
                i=30, [15, 2]
                i=31, [15, 3]
*/

// The following is the distribution among WIs in a SINGLE SG.
/*
W0 --> 0 0 0 0   1 1 1 1 ...   7 7 7 7 --> total 32 elements

wi [0,0] -> i=0, [0, 0]        wi [0,1] --> i=0, [0, 4]     wi [0,15] --> i=0, [0, 60]  | 
            i=1, [0, 1]                     i=1, [0, 5]                   i=1, [0, 61]  |               
            i=2, [0, 2]                     i=2, [0, 6]                   i=2, [0, 62]  |               
            i=3, [0, 3]                     i=3, [0, 7]                   i=3, [0, 63]  |                              

            i=4, [1, 0]                     i=4, [1, 4]                   i=4, [1, 60]  |
            i=5, [1, 1]                     i=5, [1, 5]                   i=5, [1, 61]  |
            i=6, [1, 2]                     i=6, [1, 6]                   i=6, [1, 62]  |
            i=7, [1, 3]                     i=7, [1, 7]                   i=7, [1, 63]  |
            ...                             ...                           ....          |
            i=28,[7, 0]                     i=28,[7, 4]                   i=28,[7, 60]  |
            i=29,[7, 1]                     i=29,[7, 5]                   i=29,[7, 61]  |
            i=30,[7, 2]                     i=30,[7, 6]                   i=30,[7, 62]  |
            i=31,[7, 3]                     i=31,[7, 7]                   i=31,[7, 63]  |

*/
// clang-format on

template <typename T, size_t M, size_t N>
void matrix_sum_cols(queue q, big_matrix<T, M, N> &B, nd_range<2> &r) {
  buffer<int8_t, 2> bufB(B.get_data(), range<2>(M, N));
  // size of vector is known because SG size of set by the user in this case
  int sum_cols[N] = {0};
  buffer<int> sum_cols_v(sum_cols, N); // there are total of tK/4 * 2, 16 rows
  q.submit([&](handler &cgh) {
     auto accB = bufB.get_access<access::mode::read_write>(cgh);

     auto v = sum_cols_v.get_access<access::mode::atomic>(cgh);
     auto os = sycl::stream(100000, 6144, cgh);

     cgh.parallel_for<class add_matrix>(
         r, [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();

           // TK = 32, TN = 16
           joint_matrix<sub_group, int8_t, use::b, TK, TN,
                        layout::ext_intel_packed>
               sub_b;

           joint_matrix_load(
               sg, sub_b,
               accB.template get_multi_ptr<sycl::access::decorated::no>() +
                   (global_idx * (TK / 4) * N) + sg_starty / SG_SZ * TN * 4,
               N);

           int32_t sum_local_cols[N] = {0}; // 4 local cols, N total
           // sub_b has 32x16 elements, 32 elements per WI, 4 per WI per row

           size_t
               global_index; // Index into the result array that holds the sums.

           // Keep track of cols handled in this WI
           int32_t handled_cols[N] = {-1};

           sycl::ext::intel::experimental::matrix::joint_matrix_apply(
               sg, sub_b,
               [&](int8_t &x, size_t row,
                   size_t col) { // Calculation of global index
                 int sg_idx = (int)global_idy / SG_SZ;
                 global_index = col + sg_idx * 4 /*VNNI_FACTOR*/ * SG_SZ;
                 sum_local_cols[global_index] += x;
                 handled_cols[global_index] = 1;
               });
           for (int j = 0; j < N; j++) {
             if (handled_cols[j] == 1) {
               global_index = j;
               sum_local_cols[global_index] = reduce_over_group(
                   sg, sum_local_cols[global_index], sycl::plus<>());
               atomic_fetch_add(v[global_index], sum_local_cols[global_index]);
             }
           }
         }); // parallel for
   }).wait();
  sum_cols_ref<T, M, N>(bufB.get_host_access(), sum_cols_v.get_host_access());
}

// TK = 32, TN = 16
static constexpr size_t MATRIX_K = TK / 4 * 2; // 16
static constexpr size_t MATRIX_N = TN * 4 * 2; // 128
int8_t B[MATRIX_K][MATRIX_N];

/* <    ---------------    128    ---------------------------------->
   x x x x x x x x x x x x x x x x       ..........    x x x x x x   ^
   x x x x x x x x x x x x x x x x       ..........    x x x x x x  16
   x x x x x x x x x x x x x x x x       ..........    x x x x x x   |
   .....                                                             |
   x x x x x x x x x x x x x x x x       ..........    x x x x x x   |
   x x x x x x x x x x x x x x x x       ..........    x x x x x x   v
*/
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

  matrix_sum_cols<int8_t, MATRIX_K, MATRIX_N>(q, MB, r);

  std::cout << "Passed\n";

  return 0;
}
