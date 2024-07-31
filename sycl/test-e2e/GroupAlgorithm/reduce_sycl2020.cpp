// RUN: %{build} -fsycl-device-code-split=per_kernel -I . -o %t.out
// RUN: %{run} %t.out

#include "support.h"
#include <algorithm>
#include <cassert>
#include <complex>
#include <iostream>
#include <limits>
#include <numeric>

using namespace sycl;

queue q;

template <typename SpecializationKernelName, typename InputContainer,
          class BinaryOperation>
void test(const InputContainer &input, BinaryOperation binary_op,
          typename InputContainer::value_type identity) {
  typedef typename InputContainer::value_type InputT;
  typedef InputT OutputT;
  std::array<OutputT, 6> output = {};
  constexpr OutputT init(42);
  size_t N = input.size();
  constexpr size_t G = 16;
  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for<SpecializationKernelName>(
          nd_range<1>(G, G), [=](nd_item<1> it) {
            group<1> g = it.get_group();
            auto sg = it.get_sub_group();
            int lid = it.get_local_id(0);
            out[0] = reduce_over_group(g, in[lid], binary_op);
            out[1] = reduce_over_group(g, in[lid], init, binary_op);
            out[2] = joint_reduce(g, global_ptr<const InputT>(in),
                                  global_ptr<const InputT>(in) + N, binary_op);
            out[3] =
                joint_reduce(g, global_ptr<const InputT>(in),
                             global_ptr<const InputT>(in) + N, init, binary_op);
            out[4] = joint_reduce(sg, global_ptr<const InputT>(in),
                                  global_ptr<const InputT>(in) + N, binary_op);
            out[5] =
                joint_reduce(sg, global_ptr<const InputT>(in),
                             global_ptr<const InputT>(in) + N, init, binary_op);
          });
    });
  }
  // std::reduce is not implemented yet, so use std::accumulate instead
  // TODO: use std::reduce when it will be supported
  assert(equal(output[0], std::accumulate(input.begin(), input.begin() + G,
                                          identity, binary_op)));
  assert(equal(output[1], std::accumulate(input.begin(), input.begin() + G,
                                          init, binary_op)));
  assert(equal(output[2], std::accumulate(input.begin(), input.end(), identity,
                                          binary_op)));
  assert(equal(output[3],
               std::accumulate(input.begin(), input.end(), init, binary_op)));
  assert(equal(output[4], std::accumulate(input.begin(), input.end(), identity,
                                          binary_op)));
  assert(equal(output[5],
               std::accumulate(input.begin(), input.end(), init, binary_op)));
}

int main() {
  queue q;
  if (!isSupportedDevice(q.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }

  constexpr int N = 128;
  std::array<int, N> input;
  std::iota(input.begin(), input.end(), 0);

  test<class KernelNamePlusV>(input, sycl::plus<>(), 0);
  test<class KernelNameMinimumV>(input, sycl::minimum<>(),
                                 std::numeric_limits<int>::max());
  test<class KernelNameMaximumV>(input, sycl::maximum<>(),
                                 std::numeric_limits<int>::lowest());

  test<class KernelNamePlusI>(input, sycl::plus<int>(), 0);
  test<class KernelNameMinimumI>(input, sycl::minimum<int>(),
                                 std::numeric_limits<int>::max());
  test<class KernelNameMaximumI>(input, sycl::maximum<int>(),
                                 std::numeric_limits<int>::lowest());

  test<class KernelNameMultipliesI>(input, sycl::multiplies<int>(), 1);
  test<class KernelNameBitOrI>(input, sycl::bit_or<int>(), 0);
  test<class KernelNameBitXorI>(input, sycl::bit_xor<int>(), 0);
  test<class KernelNameBitAndI>(input, sycl::bit_and<int>(), ~0);

  test<class LogicalOrInt>(input, sycl::logical_or<int>(), 0);
  test<class LogicalAndInt>(input, sycl::logical_and<int>(), 1);

  std::array<bool, N> bool_input = {};
  test<class LogicalOrBool>(bool_input, sycl::logical_or<bool>(), false);
  test<class LogicalOrVoid>(bool_input, sycl::logical_or<>(), false);
  test<class LogicalAndBool>(bool_input, sycl::logical_and<bool>(), true);
  test<class LogicalAndVoid>(bool_input, sycl::logical_and<>(), true);

  std::array<int2, N> int2_input = {};
  std::iota(int2_input.begin(), int2_input.end(), 0);
  test<class PlusInt2>(int2_input, sycl::plus<int2>(), {0, 0});
  test<class PlusInt2V>(int2_input, sycl::plus<>(), {0, 0});

  if (q.get_device().has(aspect::fp16)) {
    std::array<half, 32> half_input = {};
    std::iota(half_input.begin(), half_input.end(), 0);
    test<class PlusHalf>(half_input, sycl::plus<half>(), 0);
    test<class PlusHalfV>(half_input, sycl::plus<>(), 0);
  }

  // as part of SYCL_EXT_ONEAPI_COMPLEX_ALGORITHMS (
  // https://github.com/intel/llvm/pull/5108/ ) joint_reduce and
  // reduce_over_group now operate on std::complex limited to using the
  // sycl::plus binary operation.
#ifdef SYCL_EXT_ONEAPI_COMPLEX_ALGORITHMS
  std::array<std::complex<float>, N> input_cf;
  std::iota(input_cf.begin(), input_cf.end(), 0);
  test<class KernelNamePlusComplexF>(input_cf,
                                     sycl::plus<std::complex<float>>(), 0);
  test<class KernelNamePlusUnspecF>(input_cf, sycl::plus<>(), 0);

  if (q.get_device().has(aspect::fp64)) {
    std::array<std::complex<double>, N> input_cd;
    std::iota(input_cd.begin(), input_cd.end(), 0);
    test<class KernelNamePlusComplexD>(input_cd,
                                       sycl::plus<std::complex<double>>(), 0);
    test<class KernelNamePlusUnspecD>(input_cd, sycl::plus<>(), 0);
  } else {
    std::cout << "aspect::fp64 not supported. skipping std::complex<double>"
              << std::endl;
  }
#else
  static_assert(false, "SYCL_EXT_ONEAPI_COMPLEX_ALGORITHMS not defined");
#endif

  std::cout << "Test passed." << std::endl;
}
