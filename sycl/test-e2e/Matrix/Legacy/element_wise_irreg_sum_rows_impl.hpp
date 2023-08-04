#include <iomanip>
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
void sum_rows_ref(host_accessor<T, 2, access::mode::read> B,
                  host_accessor<int, 1, access::mode::read> sum_rows) {
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
  int sum_rows[M] = {0};
  std::cout << "M = " << VS << "\n";
  buffer<int> sum_rows_v(sum_rows, M); // there are total of tK/4 * 2, 16 rows
  q.submit([&](handler &cgh) {
     auto accB = bufB.get_access<access::mode::read_write>(cgh);

     auto v = sum_rows_v.get_access<access::mode::atomic>(cgh);
     auto os = sycl::stream{30000, 30000, cgh};
     cgh.parallel_for<class add_matrix>(
         r, [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);
           os << spmd_item << ": ";

           sycl::sub_group sg = spmd_item.get_sub_group();

           joint_matrix<T, TK, TN, matrix_layout::packed_b> sub_b(sg);

           joint_matrix_load(
               sg, sub_b,
               accB.template get_multi_ptr<access::decorated::no>() +
                   (global_idx * (TK / 4) * N) + sg_starty / SG_SZ * TN * 4,
               N, matrix_layout::packed_b);
           os << "B: ";
           auto data = sub_b.get_wi_data();
           for (int i = 0; i < data.length(); i++) {
             os << (int)data[i] << " ";
           }

           // Calculate sum of rows in sum_local_rows.
           // Depending on subgroup size we may need different local array size.
           // Implementation detail: number of matrix elements loaded in one
           // work-item element:
           static constexpr size_t pack_factor = 4;
           // size of array that we need:
           static constexpr size_t VS = (TK * TN) / (SG_SZ * pack_factor);

           // 8 local rows for SG_SZ=16; 4 local rows for SG_SZ=32; M total
           int32_t sum_local_rows[VS] = {0};

           // each WI calculates local sum of rows
           for (int row = 0; row < VS; row++) {
             for (int i = 0; i < pack_factor; i++) {
              // stopped here
               sum_local_rows[row + global_idx * (TK / 4)] +=
                   data[i + row * elems_per_wi_row];
             }
             sum_local_rows[row + global_idx * (TK / 4)] = reduce_over_group(
                 sg, sum_local_rows[row + global_idx * (TK / 4)],
                 sycl::plus<>());

             // only Groups leader perform the global reduction
             if (global_idy % SG_SZ == 0) {
               atomic_fetch_add(v[row + global_idx * (TK / 4)],
                                sum_local_rows[row + global_idx * (TK / 4)]);
             }
           }
           os << "\n";
         }); // parallel for
   }).wait();
  sum_rows_ref<T, M, N>(bufB.get_host_access(read_only),
                        sum_rows_v.get_host_access(read_only));
}

static constexpr size_t MATRIX_K = TK / 4 * 2;
static constexpr size_t MATRIX_N = TN * 4 * 2;
int8_t B[MATRIX_K][MATRIX_N];

int main() {
  std::cout << "Matrix B: " << TK << "x" << TN << "\n";
  std::cout << "Matrix B vnni: " << MATRIX_K << "x" << MATRIX_N << "\n";
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
  for (int i = 0; i < MATRIX_K; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      std::cout << std::setw(2) << (int)B[i][j] << " ";
    }
    std::cout << "\n";
  }

  matrix_sum_rows<int8_t, MATRIX_K, MATRIX_N>(q, MB, r);

  return 0;
}
