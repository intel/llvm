// UNSUPPORTED: hip
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-device-code-split=per_kernel %s -I . -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include "support.h"
#include <algorithm>
#include <cassert>
#include <complex>
#include <limits>
#include <numeric>
#include <sycl/sycl.hpp>
using namespace sycl;

template <typename SpecializationKernelName, typename InputContainer,
          typename OutputContainer, class BinaryOperation>
void test(queue q, InputContainer input, OutputContainer output,
          BinaryOperation binary_op,
          typename OutputContainer::value_type identity) {
  typedef typename InputContainer::value_type InputT;
  typedef typename OutputContainer::value_type OutputT;
  constexpr OutputT init = 42;
  constexpr size_t N = input.size();
  constexpr size_t G = 64;
  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for<SpecializationKernelName>(
          nd_range<1>(G, G), [=](nd_item<1> it) {
            group<1> g = it.get_group();
            int lid = it.get_local_id(0);
            out[0] = reduce_over_group(g, in[lid], binary_op);
            out[1] = reduce_over_group(g, in[lid], init, binary_op);
            out[2] = joint_reduce(g, in.get_pointer(), in.get_pointer() + N,
                                  binary_op);
            out[3] = joint_reduce(g, in.get_pointer(), in.get_pointer() + N,
                                  init, binary_op);
          });
    });
  }
  // std::reduce is not implemented yet, so use std::accumulate instead
  // TODO: use std::reduce when it will be supported
  assert(output[0] == std::accumulate(input.begin(), input.begin() + G,
                                      identity, binary_op));
  assert(output[1] ==
         std::accumulate(input.begin(), input.begin() + G, init, binary_op));
  assert(output[2] ==
         std::accumulate(input.begin(), input.end(), identity, binary_op));
  assert(output[3] ==
         std::accumulate(input.begin(), input.end(), init, binary_op));
}

int main() {
  queue q;
  if (!isSupportedDevice(q.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }

  constexpr int N = 128;
  std::array<int, N> input;
  std::array<int, 4> output;
  std::iota(input.begin(), input.end(), 0);
  std::fill(output.begin(), output.end(), 0);

  test<class KernelNamePlusV>(q, input, output, sycl::plus<>(), 0);
  test<class KernelNameMinimumV>(q, input, output, sycl::minimum<>(),
                                 std::numeric_limits<int>::max());
  test<class KernelNameMaximumV>(q, input, output, sycl::maximum<>(),
                                 std::numeric_limits<int>::lowest());

  test<class KernelNamePlusI>(q, input, output, sycl::plus<int>(), 0);
  test<class KernelNameMinimumI>(q, input, output, sycl::minimum<int>(),
                                 std::numeric_limits<int>::max());
  test<class KernelNameMaximumI>(q, input, output, sycl::maximum<int>(),
                                 std::numeric_limits<int>::lowest());

  test<class KernelNameMultipliesI>(q, input, output, sycl::multiplies<int>(),
                                    1);
  test<class KernelNameBitOrI>(q, input, output, sycl::bit_or<int>(), 0);
  test<class KernelNameBitXorI>(q, input, output, sycl::bit_xor<int>(), 0);
  test<class KernelNameBitAndI>(q, input, output, sycl::bit_and<int>(), ~0);

  // as part of SYCL_EXT_ONEAPI_COMPLEX_ALGORITHMS (
  // https://github.com/intel/llvm/pull/5108/ ) joint_reduce and
  // reduce_over_group now operate on std::complex limited to using the
  // sycl::plus binary operation.
#ifdef SYCL_EXT_ONEAPI_COMPLEX_ALGORITHMS
  std::array<std::complex<float>, N> input_cf;
  std::array<std::complex<float>, 4> output_cf;
  std::iota(input_cf.begin(), input_cf.end(), 0);
  std::fill(output_cf.begin(), output_cf.end(), 0);
  test<class KernelNamePlusComplexF>(q, input_cf, output_cf,
                                     sycl::plus<std::complex<float>>(), 0);
  test<class KernelNamePlusUnspecF>(q, input_cf, output_cf, sycl::plus<>(), 0);

  if (q.get_device().has(aspect::fp64)) {
    std::array<std::complex<double>, N> input_cd;
    std::array<std::complex<double>, 4> output_cd;
    std::iota(input_cd.begin(), input_cd.end(), 0);
    std::fill(output_cd.begin(), output_cd.end(), 0);
    test<class KernelNamePlusComplexD>(q, input_cd, output_cd,
                                       sycl::plus<std::complex<double>>(), 0);
    test<class KernelNamePlusUnspecD>(q, input_cd, output_cd, sycl::plus<>(),
                                      0);
  } else {
    std::cout << "aspect::fp64 not supported. skipping std::complex<double>"
              << std::endl;
  }
#else
  static_assert(false, "SYCL_EXT_ONEAPI_COMPLEX_ALGORITHMS not defined");
#endif

  std::cout << "Test passed." << std::endl;
}
