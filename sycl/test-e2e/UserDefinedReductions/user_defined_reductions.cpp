// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// UNSUPPORTED: cuda || hip

#include <complex>
#include <numeric>

#include <sycl/ext/oneapi/experimental/user_defined_reductions.hpp>
#include <sycl/sycl.hpp>

template <typename T> struct UserDefinedSum {
  T operator()(T a, T b) { return a + b; }
};

template <typename T> struct UserDefinedMax {
  T operator()(T a, T b) { return (a < b) ? b : a; }
};

template <typename T> struct UserDefinedBitAnd {
  T operator()(const T &a, const T &b) const { return a & b; }
};

template <typename T> struct UserDefinedMultiplies {
  T operator()(const T &a, const T &b) const { return a * b; }
};

struct custom_type_nested {
  static constexpr int default_i_value = 6;
  static constexpr float default_f_value = 1.5;

  constexpr custom_type_nested() = default;
  constexpr custom_type_nested(int i, float f) : i(i), f(f) {}

  int i = default_i_value;
  float f = default_f_value;
};

inline bool operator==(const custom_type_nested &lhs,
                       const custom_type_nested &rhs) {
  return lhs.i == rhs.i && lhs.f == rhs.f;
}

inline std::ostream &operator<<(std::ostream &out,
                                const custom_type_nested &v) {
  return out << "custom_type_nested { .i = " << v.i << ", .f = " << v.f << "}";
}

struct custom_type {
  static constexpr unsigned long long default_ull_value = 42;

  constexpr custom_type() = default;
  constexpr custom_type(int i, float f, unsigned long long ull)
      : n(i, f), ull(ull) {}

  custom_type_nested n;
  unsigned long long ull = default_ull_value;
};

inline bool operator==(const custom_type &lhs, const custom_type &rhs) {
  return lhs.n == rhs.n && lhs.ull == rhs.ull;
}

inline custom_type operator+(const custom_type &lhs, const custom_type &rhs) {
  return custom_type(lhs.n.i + rhs.n.i, lhs.n.f + rhs.n.f, lhs.ull + rhs.ull);
}

inline std::ostream &operator<<(std::ostream &os, const custom_type &v) {
  os << "custom_type { .n = " << v.n << ", .ull = " << v.ull << "}";
  return os;
}

struct custom_type_wo_default_ctor {
  static constexpr unsigned long long default_ull_value = 42;

  constexpr custom_type_wo_default_ctor() = delete;
  constexpr custom_type_wo_default_ctor(int i, float f, unsigned long long ull)
      : n(i, f), ull(ull) {}

  custom_type_nested n;
  unsigned long long ull = default_ull_value;
};

inline bool operator==(const custom_type_wo_default_ctor &lhs,
                       const custom_type_wo_default_ctor &rhs) {
  return lhs.n == rhs.n && lhs.ull == rhs.ull;
}

inline custom_type_wo_default_ctor
operator+(const custom_type_wo_default_ctor &lhs,
          const custom_type_wo_default_ctor &rhs) {
  return custom_type_wo_default_ctor(lhs.n.i + rhs.n.i, lhs.n.f + rhs.n.f,
                                     lhs.ull + rhs.ull);
}

template <typename T, std::size_t... Is>
constexpr std::array<T, sizeof...(Is)> init_array(T value,
                                                  std::index_sequence<Is...>) {
  return {{(static_cast<void>(Is), value)...}};
}

inline std::ostream &operator<<(std::ostream &os,
                                const custom_type_wo_default_ctor &v) {
  os << "custom_type_wo_default_ctor { .n = " << v.n << ", .ull = " << v.ull
     << "}";
  return os;
}

using namespace sycl;

template <typename InputContainer, typename OutputContainer,
          class BinaryOperation>
void test(queue q, InputContainer input, OutputContainer output,
          BinaryOperation binary_op, size_t workgroup_size,
          typename OutputContainer::value_type identity,
          typename OutputContainer::value_type init) {
  using InputT = typename InputContainer::value_type;
  using OutputT = typename OutputContainer::value_type;
  constexpr size_t N = input.size();
  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};

      size_t temp_memory_size = workgroup_size * sizeof(InputT);
      auto scratch = sycl::local_accessor<std::byte, 1>(temp_memory_size, cgh);
      cgh.parallel_for(
          nd_range<1>(workgroup_size, workgroup_size), [=](nd_item<1> it) {
            // Create a handle that associates the group with an allocation it
            // can use
            auto handle =
                sycl::ext::oneapi::experimental::group_with_scratchpad(
                    it.get_group(), sycl::span(&scratch[0], temp_memory_size));

            const InputT *first =
                in.template get_multi_ptr<access::decorated::no>();
            const InputT *last = first + N;
            // check reduce_over_group w/o init
            out[0] = sycl::ext::oneapi::experimental::reduce_over_group(
                handle, in[it.get_global_id(0)], binary_op);

            // check reduce_over_group with init
            out[1] = sycl::ext::oneapi::experimental::reduce_over_group(
                handle, in[it.get_global_id(0)], init, binary_op);

            // check joint_reduce w/o init
            out[2] = sycl::ext::oneapi::experimental::joint_reduce(
                handle, first, last, binary_op);

            // check joint_reduce with init
            out[3] = sycl::ext::oneapi::experimental::joint_reduce(
                handle, first, last, init, binary_op);
          });
    });
    q.wait();
  }
  assert(output[0] == std::accumulate(input.begin(),
                                      input.begin() + workgroup_size, identity,
                                      binary_op));
  assert(output[1] == std::accumulate(input.begin(),
                                      input.begin() + workgroup_size, init,
                                      binary_op));
  assert(output[2] ==
         std::accumulate(input.begin(), input.end(), identity, binary_op));
  assert(output[3] ==
         std::accumulate(input.begin(), input.end(), init, binary_op));
}

int main() {
  queue q;

  constexpr int N = 128;
  std::array<int, N> input;
  std::array<int, 4> output;
  std::iota(input.begin(), input.end(), 0);
  std::fill(output.begin(), output.end(), 0);

  // queue, input array, output array, binary_op, WG size, identity, init
  test(q, input, output, UserDefinedSum<int>{}, 64, 0, 42);
  test(q, input, output, UserDefinedSum<int>{}, 32, 0, 42);
  test(q, input, output, UserDefinedSum<int>{}, 5, 0, 42);
  test(q, input, output, UserDefinedMax<int>{}, 64,
       std::numeric_limits<int>::lowest(), 42);
  test(q, input, output, UserDefinedMultiplies<int>(), 64, 1, 42);
  test(q, input, output, UserDefinedBitAnd<int>{}, 64, ~0, 42);

  test(q, input, output, sycl::plus<int>(), 64, 0, 42);
  test(q, input, output, sycl::maximum<>(), 64,
       std::numeric_limits<int>::lowest(), 42);
  test(q, input, output, sycl::minimum<int>(), 64,
       std::numeric_limits<int>::max(), 42);

  test(q, input, output, sycl::multiplies<int>(), 64, 1, 42);
  test(q, input, output, sycl::bit_or<int>(), 64, 0, 42);
  test(q, input, output, sycl::bit_xor<int>(), 64, 0, 42);
  test(q, input, output, sycl::bit_and<int>(), 64, ~0, 42);

  std::array<custom_type, N> input_custom;
  std::array<custom_type, 4> output_custom;
  test(q, input_custom, output_custom, UserDefinedSum<custom_type>{}, 64,
       custom_type(0, 0., 0), custom_type(10, 0., 5));

  custom_type_wo_default_ctor value(1, 2.5, 3);
  std::array<custom_type_wo_default_ctor, N> input_custom_wo_default_ctor =
      init_array(value, std::make_index_sequence<N>());
  std::array<custom_type_wo_default_ctor, 4> output_custom_wo_default_ctor =
      init_array(value, std::make_index_sequence<4>());
  test(q, input_custom_wo_default_ctor, output_custom_wo_default_ctor,
       UserDefinedSum<custom_type_wo_default_ctor>{}, 64,
       custom_type_wo_default_ctor(0, 0., 0),
       custom_type_wo_default_ctor(10, 0., 5));

#ifdef SYCL_EXT_ONEAPI_COMPLEX_ALGORITHMS
  std::array<std::complex<float>, N> input_cf;
  std::array<std::complex<float>, 4> output_cf;
  std::iota(input_cf.begin(), input_cf.end(), 0);
  std::fill(output_cf.begin(), output_cf.end(), 0);
  test(q, input_cf, output_cf, sycl::plus<std::complex<float>>(), 64, 0, 42);
  test(q, input_cf, output_cf, sycl::plus<>(), 64, 0, 42);
#else
  static_assert(false, "SYCL_EXT_ONEAPI_COMPLEX_ALGORITHMS not defined");
#endif

  return 0;
}
