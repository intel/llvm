static constexpr size_t M_MULTIPLIER = 16;

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

template <typename T, size_t M, size_t N>
void assert_ops_ref(host_accessor<T, 2, access::mode::read_write> C,
                    const T ref) {
  for (size_t i = 0; i < M; i++)
    for (size_t j = 0; j < N; j++) {
      if (std::is_same_v<T, bfloat16>) {
        auto diff = make_fp32(C[i][j]) - make_fp32(ref);
        assert(std::fabs(static_cast<float>(diff)) <
               std::numeric_limits<float>::epsilon());
      } else if (std::is_same_v<T, int8_t>) {
        assert(C[i][j] == ref);
      }
    }
}

template <typename T, typename T1, size_t TM, size_t TK>
void matrix_verify_add(const T1 val1, const T1 val2, const T1 result) {
  static constexpr size_t M = TM * M_MULTIPLIER;
  static constexpr size_t K = 128;
  T MatA[M][K];

  size_t NDRangeM = M / TM;
  size_t NDRangeK = K / TK;
  queue q;
  nd_range<2> r({NDRangeM, NDRangeK * SG_SZ}, {1, 1 * SG_SZ});
  big_matrix<T, M, K> A((T *)&MatA);

  buffer<T, 2> bufA(A.get_data(), range<2>(M, K));

  q.submit([&](handler &cgh) {
     sycl::accessor accA{bufA, cgh, sycl::read_write};

     cgh.parallel_for(r, [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(
                             SG_SZ)]] {
       const auto global_idx = spmd_item.get_global_id(0);
       const auto global_idy = spmd_item.get_global_id(1);
       const auto sg_startx = global_idx - spmd_item.get_local_id(0);
       const auto sg_starty = global_idy - spmd_item.get_local_id(1);

       sub_group sg = spmd_item.get_sub_group();
       joint_matrix<sub_group, T, use::a, TM, TK, layout::row_major> sub_a;

       joint_matrix_fill(sg, sub_a, val1);

       auto wi_slice_a =
           sycl::ext::intel::experimental::matrix::get_wi_data(sg, sub_a);
       for (int i = 0; i < wi_slice_a.length(); i++) {
         wi_slice_a[i] = wi_slice_a[i] + val2;
       }

       ext::intel::experimental::matrix::joint_matrix_store(
           sg, sub_a,
           accA.get_pointer() + (sg_startx * TM) * K + sg_starty / SG_SZ * TK,
           K);
     }); // parallel for
   }).wait();
  assert_ops_ref<T, M, K>(bufA.get_host_access(), result);
}

template <typename Ta, size_t tM, size_t tK> void add_ref() {
  if constexpr (std::is_same_v<Ta, bfloat16>) {
    // Tests whether 5 + 2 = 7 operation is successful.
    matrix_verify_add<bfloat16, bfloat16, tM, tK>(bfloat16(5.0), bfloat16(2.0),
                                                  bfloat16(7.0));
  }
  if constexpr (std::is_same_v<Ta, int8_t>) {
    matrix_verify_add<int8_t, int, tM, tK>(5 /*val1*/, 2 /*val2*/,
                                           7 /*result*/);
  }
}

int main() {
  add_ref<bfloat16, 1 /*TM*/, 16 /*TK*/>();
  add_ref<bfloat16, 2 /*TM*/, 16 /*TK*/>();
  add_ref<bfloat16, 3 /*TM*/, 16 /*TK*/>();
  add_ref<bfloat16, 4 /*TM*/, 16 /*TK*/>();
  add_ref<bfloat16, 5 /*TM*/, 16 /*TK*/>();
  add_ref<bfloat16, 6 /*TM*/, 16 /*TK*/>();
  add_ref<bfloat16, 7 /*TM*/, 16 /*TK*/>();

  add_ref<int8_t, 1 /*TM*/, 32 /*TK*/>();
  add_ref<int8_t, 2 /*TM*/, 32 /*TK*/>();
  add_ref<int8_t, 3 /*TM*/, 32 /*TK*/>();
  add_ref<int8_t, 4 /*TM*/, 32 /*TK*/>();
  add_ref<int8_t, 5 /*TM*/, 32 /*TK*/>();
  add_ref<int8_t, 6 /*TM*/, 32 /*TK*/>();
  add_ref<int8_t, 7 /*TM*/, 32 /*TK*/>();

  std::cout << "Passed\n";
}
