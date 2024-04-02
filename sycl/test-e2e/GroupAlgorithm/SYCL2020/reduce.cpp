// RUN: %{build} -I . -o %t.out
// RUN: %{run} %t.out

#include "support.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <numeric>
#include <sycl/detail/core.hpp>
#include <sycl/group_algorithm.hpp>
using namespace sycl;

template <typename SpecializationKernelName, typename InputContainer,
          typename OutputContainer, class BinaryOperation>
void test(queue q, InputContainer input, OutputContainer output,
          BinaryOperation binary_op,
          typename OutputContainer::value_type identity) {
  typedef typename InputContainer::value_type InputT;
  typedef typename OutputContainer::value_type OutputT;
  OutputT init = 42;
  size_t N = input.size();
  size_t G = 64;
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
            out[2] = joint_reduce(
                g, in.template get_multi_ptr<access::decorated::no>(),
                in.template get_multi_ptr<access::decorated::no>() + N,
                binary_op);
            out[3] = joint_reduce(
                g, in.template get_multi_ptr<access::decorated::no>(),
                in.template get_multi_ptr<access::decorated::no>() + N, init,
                binary_op);
          });
    });
  }
  // std::reduce is not implemented yet, so use std::accumulate instead
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

  test<class KernelName_WonwuUVPUPOTKRKIBtT>(q, input, output,
                                             sycl::multiplies<int>(), 1);
  test<class KernelName_qYBaJDZTMGkdIwD>(q, input, output, sycl::bit_or<int>(),
                                         0);
  test<class KernelName_eLSFt>(q, input, output, sycl::bit_xor<int>(), 0);
  test<class KernelName_uFhJnxSVhNAiFPTG>(q, input, output,
                                          sycl::bit_and<int>(), ~0);

  std::cout << "Test passed." << std::endl;
}
