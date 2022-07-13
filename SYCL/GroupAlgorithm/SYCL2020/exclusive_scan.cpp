// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -I . -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// disabling hip because some of the binary_ops tested are not supported
// getting undefined symbols for a handful of __spirv__ * functions.
// XFAIL: hip

#include "support.h"
#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>
#include <sycl/sycl.hpp>
#include <vector>
using namespace sycl;

template <class SpecializationKernelName, int TestNumber>
class exclusive_scan_kernel;

// std::exclusive_scan isn't implemented yet, so use serial implementation
// instead
namespace emu {
template <typename InputIterator, typename OutputIterator,
          class BinaryOperation, typename T>
OutputIterator exclusive_scan(InputIterator first, InputIterator last,
                              OutputIterator result, T init,
                              BinaryOperation binary_op) {
  T partial = init;
  for (InputIterator it = first; it != last; ++it) {
    *(result++) = partial;
    partial = binary_op(partial, *it);
  }
  return result;
}
} // namespace emu

template <typename SpecializationKernelName, typename InputContainer,
          typename OutputContainer, class BinaryOperation>
void test(queue q, InputContainer input, OutputContainer output,
          BinaryOperation binary_op,
          typename OutputContainer::value_type identity) {
  typedef typename InputContainer::value_type InputT;
  typedef typename OutputContainer::value_type OutputT;
  typedef class exclusive_scan_kernel<SpecializationKernelName, 0> kernel_name0;
  typedef class exclusive_scan_kernel<SpecializationKernelName, 1> kernel_name1;
  typedef class exclusive_scan_kernel<SpecializationKernelName, 2> kernel_name2;
  typedef class exclusive_scan_kernel<SpecializationKernelName, 3> kernel_name3;
  OutputT init = 42;
  size_t N = input.size();
  size_t G = 64;
  std::vector<OutputT> expected(N);
  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for<kernel_name0>(nd_range<1>(G, G), [=](nd_item<1> it) {
        group<1> g = it.get_group();
        int lid = it.get_local_id(0);
        out[lid] = exclusive_scan_over_group(g, in[lid], binary_op);
      });
    });
  }
  emu::exclusive_scan(input.begin(), input.begin() + G, expected.begin(),
                      identity, binary_op);
  assert(std::equal(output.begin(), output.begin() + G, expected.begin()));

  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for<kernel_name1>(nd_range<1>(G, G), [=](nd_item<1> it) {
        group<1> g = it.get_group();
        int lid = it.get_local_id(0);
        out[lid] = exclusive_scan_over_group(g, in[lid], init, binary_op);
      });
    });
  }
  emu::exclusive_scan(input.begin(), input.begin() + G, expected.begin(), init,
                      binary_op);
  assert(std::equal(output.begin(), output.begin() + G, expected.begin()));

  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for<kernel_name2>(nd_range<1>(G, G), [=](nd_item<1> it) {
        group<1> g = it.get_group();
        joint_exclusive_scan(g, in.get_pointer(), in.get_pointer() + N,
                             out.get_pointer(), binary_op);
      });
    });
  }
  emu::exclusive_scan(input.begin(), input.begin() + N, expected.begin(),
                      identity, binary_op);
  assert(std::equal(output.begin(), output.begin() + N, expected.begin()));

  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for<kernel_name3>(nd_range<1>(G, G), [=](nd_item<1> it) {
        group<1> g = it.get_group();
        joint_exclusive_scan(g, in.get_pointer(), in.get_pointer() + N,
                             out.get_pointer(), init, binary_op);
      });
    });
  }
  emu::exclusive_scan(input.begin(), input.begin() + N, expected.begin(), init,
                      binary_op);
  assert(std::equal(output.begin(), output.begin() + N, expected.begin()));
}

int main() {
  queue q;
  if (!isSupportedDevice(q.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }

  constexpr int N = 128;
  std::array<int, N> input;
  std::array<int, N> output;
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
  test<class KernelName_VzAPutpBRRJrQPB>(q, input, output,
                                         sycl::multiplies<int>(), 1);
  test<class KernelName_UXdGbr>(q, input, output, sycl::bit_or<int>(), 0);
  test<class KernelName_saYaodNyJknrPW>(q, input, output, sycl::bit_xor<int>(),
                                        0);
  test<class KernelName_GPcuAlvAOjrDyP>(q, input, output, sycl::bit_and<int>(),
                                        ~0);

  std::cout << "Test passed." << std::endl;
}
