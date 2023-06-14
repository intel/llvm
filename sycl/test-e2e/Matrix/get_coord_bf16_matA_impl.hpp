#define TM 8
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

template <typename T, size_t M, size_t K>
void sum_rows_ref(host_accessor<T, 2, access::mode::read_write> A,
                  host_accessor<int, 1, access::mode::read_write> sum_rows) {
  int sum_rows_ref[M] = {0};
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < K; j++) {
      sum_rows_ref[i] += A[i][j];
    }
    auto diff = sum_rows[i] - sum_rows_ref[i];
    assert(std::fabs(static_cast<int>(diff)) <=
           std::numeric_limits<int>::epsilon());
  }
}

// clang-format off
/*
Here's how the data is distributed among work items
0 0  0 0
/
/
1 1  1 1
/
/
2 2  2 2
/
/ 
3 3  3 3 
W0 --> 0 0 1 1 2 2 3 3 .... 7 7
wi [0,0] -> i=0, [0, 0]        wi [0,1] --> i=0, [0, 2]     wi [0,15] --> i=0, [0, 30]
            i=1, [0, 1]                     i=1, [0, 3]                   i=1, [0, 31]
            i=2, [1, 0]                     i=2, [1, 2]                   i=2, [1, 30]
            i=3, [1, 1]                     i=3, [1, 3]                   i=3, [1, 31]
            i=4, [2, 0]                     i=4, [2, 2]                   ...
            i=5, [2, 1]                     i=5, [2, 3]
            ...                             ....
            i=14,[7, 0]                     i=14, [7, 2]
            i=15,[7, 1]                     i=15, [7, 3]                  i=15, [7, 31]
*/
//clang-format on
std::tuple<uint32_t, uint32_t> get_coord_ref(int i, int wi_number) {
  return std::make_tuple(i/2, ((i%2) + (wi_number*2)));
}

//clang-format off
/* 
Here's how the distribution of the A matrix looks like for this test case
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
<---------------------------------  SG1 --------------------------------->
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
x x x x  x x x x    x x x x  x x x x  x x x x  x x x x    x x x x  x x x x
<0> <1>  <2> <3>    <4> <5>  <6> <7>  ..... WORK ITEMS
Each work item has 16 elements <8 rows and 2 cols of the original matrix>
the data_slice in holds the matrix elements in the following order:
0 0  0 0
   /
  /
1 1  1 1
   /
  /
2 2  2 2
  /
 / 
3 3  3 3 
W0 --> 0 0 1 1 2 2 3 3 .... 7 7
*/
//clang-format on
template <typename T, size_t M, size_t K>
void matrix_sum_rows(queue q, big_matrix<T, M, K> &A, nd_range<2> &r) {
  buffer<int8_t, 2> bufA(A.get_data(), range<2>(M, K));
  // size of vector is known because SG size of set by the user in this case
  int sum_rows[M] = {0};
  buffer<int> sum_rows_v(sum_rows, M); // there are total of M rows
  q.submit([&](handler &cgh) {
     auto accA = bufA.get_access<access::mode::read_write>(cgh);

     auto v = sum_rows_v.get_access<access::mode::atomic>(cgh);
     auto os = sycl::stream(100000, 6144, cgh);

     cgh.parallel_for<class add_matrix>(
         r, [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           ext::oneapi::sub_group sg = spmd_item.get_sub_group();

           // TM = 8, TK = 32
          joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major>
               sub_a;

          joint_matrix_load(
                 sg, sub_a, accA.template get_multi_ptr<access::decorated::no>() + (global_idx * TM * K) + TK,
                 K); 

           // calculate sum of rows in sum_rows_v[8], there are 8 rows in sub_a
           int32_t sum_local_rows[M] = {0}; // 8 local rows, M total
           // sub_a has 8x32 elements, 16 elements per WI, 2 per WI per row
          auto data = sycl::ext::intel::experimental::matrix::get_wi_data(sg, sub_a);

           size_t global_index; // Index into the result array that holds the sums.
          
          // Keep track of rows handled in this WI
          int32_t handled_rows[M] = {-1};

          //  each WI calculates local sum of rows
          for (int i = 0; i < data.length(); ++i) {
            // get the index of the element in the submatrix
            auto data_item = data[i];
            auto [row, col] = data_item.get_coord();
            global_index = row + global_idx*TM;

            sum_local_rows[global_index] += data[i];

            handled_rows[global_index] = 1;
          }
          
          for (int j=0; j < M; j++) {
              if (handled_rows[j] == 1) {
                global_index = j;
                sum_local_rows[global_index] = reduce_over_group(
                    sg, sum_local_rows[global_index],
                    sycl::plus<>());
                // only Groups leader perform the global reduction
                if (global_idy % SG_SZ == 0) {
                  atomic_fetch_add(v[global_index],
                                  sum_local_rows[global_index]);
                }
              }
          } 
         }); // parallel for
   }).wait();
  sum_rows_ref<T, M, K>(bufA.get_host_access(), sum_rows_v.get_host_access());
}


static constexpr size_t MATRIX_M = TM * 2;
static constexpr size_t MATRIX_K = TK * 2;
int8_t A[MATRIX_M][MATRIX_K];

int main() {
  big_matrix<int8_t, MATRIX_M, MATRIX_K> MA((int8_t *)&A);

  size_t NDRangeM = MATRIX_M / TM;
  size_t NDRangeK = MATRIX_K / TK;
  queue q;
  nd_range<2> r({NDRangeM, NDRangeK * SG_SZ}, {1, 1 * SG_SZ});

  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_K; j++) {
      A[i][j] = i;
    }
  }

  matrix_sum_rows<int8_t, MATRIX_M, MATRIX_K>(q, MA, r);

  std::cout << "Passed\n";

  return 0;
}
