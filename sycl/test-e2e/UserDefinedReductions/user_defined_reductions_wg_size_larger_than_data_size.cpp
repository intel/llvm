// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <iostream>
#include <numeric>

#include <sycl/ext/oneapi/experimental/user_defined_reductions.hpp>
#include <sycl/sycl.hpp>

// 1. Allocate an buffer of 16 elements where first 8 elements filled with 1,
//    ..., 8 and the second 8 elements filled with 0.
// 2. Submit a kernel with one wg of size 16.
// 3. invoke joint_reduce with first ==  start of the buffer and last ==  start
//    of the buffer + 8 elems.
// 4. The result should be equal to 1, as 1 is the minimum number in the
//    selection.

template <typename T = void> struct UserDefinedMinimum {
  T operator()(const T &lhs, const T &rhs) const {
    return std::less<T>()(lhs, rhs) ? lhs : rhs;
  }
};

constexpr int segment_size = 8;

using namespace sycl;

template <typename InputContainer, typename OutputContainer,
          class BinaryOperation>
void test(queue q, InputContainer input, OutputContainer output,
          BinaryOperation binary_op, size_t workgroup_size,
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
            InputT *segment_begin = in.get_pointer();
            InputT *segment_end = in.get_pointer() + segment_size;
            auto handle =
                sycl::ext::oneapi::experimental::group_with_scratchpad(
                    it.get_group(), sycl::span(&scratch[0], temp_memory_size));
            OutputT group_aggregate =
                sycl::ext::oneapi::experimental::joint_reduce(
                    handle, segment_begin, segment_end, init, binary_op);
            if (it.get_local_linear_id() == 0) {
              out[it.get_group_linear_id()] = group_aggregate;
            }
          });
    });
    q.wait();
  }
  assert(output[0] == 1);
}

int main() {
  queue q;

  constexpr int N = 16;
  std::array<int, N> input;
  std::iota(input.begin(), input.begin() + segment_size, 1);
  std::fill(input.begin() + segment_size, input.end(), 0);
  std::array<int, 1> output;

  // queue, input array, output array, binary_op, segment_size, WG size, init
  test(q, input, output, sycl::minimum<int>{}, N, INT_MAX);
  test(q, input, output, UserDefinedMinimum<int>{}, N, INT_MAX);
  return 0;
}
