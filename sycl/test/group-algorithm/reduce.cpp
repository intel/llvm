// UNSUPPORTED: cuda
// OpenCL C 2.x alike work-group functions not yet supported by CUDA.
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>
using namespace sycl;
using namespace sycl::intel;

template <typename SpecializationKernelName, typename InputContainer,
          typename OutputContainer, class BinaryOperation>
void test(queue q, InputContainer input, OutputContainer output,
          BinaryOperation binary_op,
          typename OutputContainer::value_type identity) {
  typedef typename InputContainer::value_type InputT;
  typedef typename OutputContainer::value_type OutputT;
  OutputT init = 42;
  size_t N = input.size();
  size_t G = 16;
  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto in = in_buf.template get_access<access::mode::read>(cgh);
      auto out = out_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for<SpecializationKernelName>(nd_range<1>(G, G), [=](nd_item<1> it) {
        group<1> g = it.get_group();
        int lid = it.get_local_id(0);
        out[0] = reduce(g, in[lid], binary_op);
        out[1] = reduce(g, in[lid], init, binary_op);
        out[2] = reduce(g, in.get_pointer(), in.get_pointer() + N, binary_op);
        out[3] =
            reduce(g, in.get_pointer(), in.get_pointer() + N, init, binary_op);
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
  std::string version = q.get_device().get_info<info::device::version>();
  if (version < std::string("2.0")) {
    std::cout << "Skipping test\n";
    return 0;
  }

  constexpr int N = 32;
  std::array<int, N> input;
  std::array<int, 4> output;
  std::iota(input.begin(), input.end(), 0);
  std::fill(output.begin(), output.end(), 0);

#if __cplusplus >= 201402L
  test<class KernelNamePlusV>(q, input, output, plus<>(), 0);
  test<class KernelNameMinimumV>(q, input, output, minimum<>(), std::numeric_limits<int>::max());
  test<class KernelNameMaximumV>(q, input, output, maximum<>(), std::numeric_limits<int>::lowest());
#endif
  test<class KernelNamePlusI>(q, input, output, plus<int>(), 0);
  test<class KernelNameMinimumI>(q, input, output, minimum<int>(), std::numeric_limits<int>::max());
  test<class KernelNameMaximumI>(q, input, output, maximum<int>(), std::numeric_limits<int>::lowest());

  std::cout << "Test passed." << std::endl;
}
