
#define TM 8
#define TN SG_SZ
#define TK 16

static float make_fp32(bfloat16 x) {
  unsigned int y = sycl::bit_cast<uint16_t>(x);
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

template <typename T> struct apply_add {
  void operator()(T &x) const { x = x + bfloat16(2); }
};

template <typename T, size_t M, size_t N, typename F>
void matrix_verify_add(queue q, big_matrix<T, M, N> &A, nd_range<2> &r,
                       const float ref, F &&lambda) {
  buffer<bfloat16, 2> bufA(A.get_data(), range<2>(M, N));

  q.submit([&](handler &cgh) {
     accessor accA{bufA, cgh};

     cgh.parallel_for(r, [accA, lambda](
                             nd_item<2> spmd_item) [[intel::reqd_sub_group_size(
                             SG_SZ)]] {
       const auto global_idx = spmd_item.get_global_id(0);
       const auto global_idy = spmd_item.get_global_id(1);
       const auto sg_startx = global_idx - spmd_item.get_local_id(0);
       const auto sg_starty = global_idy - spmd_item.get_local_id(1);

       sub_group sg = spmd_item.get_sub_group();
       joint_matrix<sub_group, T, use::a, TM, TK, layout::row_major> sub_a;

       joint_matrix_fill(sg, sub_a, bfloat16(5.0));

       joint_matrix_apply(sg, sub_a, lambda);

       ext::intel::experimental::matrix::joint_matrix_store(
           sg, sub_a,
           accA.template get_multi_ptr<access::decorated::no>() +
               (sg_startx * TM) * N + sg_starty / SG_SZ * TN,
           N);
     }); // parallel for
   }).wait();
  // Check if the results are correct
  {
    host_accessor Acc{bufA};
    assert(std::all_of(Acc.begin(), Acc.end(), [=](auto Elem) {
      return (std::fabs(static_cast<float>(make_fp32(Elem) - ref)) <
              std::numeric_limits<float>::epsilon());
    }));
  }
}

static constexpr size_t MATRIX_M = TM * 2;
static constexpr size_t MATRIX_N = TN * 2;
bfloat16 A[MATRIX_M][MATRIX_N];
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
  big_matrix<bfloat16, MATRIX_M, MATRIX_N> MA((bfloat16 *)&A);

  size_t NDRangeM = MATRIX_M / TM;
  size_t NDRangeN = MATRIX_N / TN;
  queue q;
  nd_range<2> r({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ});

  matrix_verify_add<bfloat16, MATRIX_M, MATRIX_N>(
      q, MA, r, 7.0, [=](bfloat16 &x) { x = x + bfloat16(2); });
  matrix_verify_add<bfloat16, MATRIX_M, MATRIX_N>(q, MA, r, 7.0,
                                                  apply_add<bfloat16>());
  std::cout << "Passed\n";
  return 0;
}
