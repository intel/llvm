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
#include <vector>
using namespace sycl;
using namespace sycl::intel;

template <class BinaryOperation, int TestNumber>
class inclusive_scan_kernel;

// std::inclusive_scan isn't implemented yet, so use serial implementation
// instead
namespace emu {
template <typename InputIterator, typename OutputIterator,
          class BinaryOperation, typename T>
OutputIterator inclusive_scan(InputIterator first, InputIterator last,
                              OutputIterator result, BinaryOperation binary_op,
                              T init) {
  T partial = init;
  for (InputIterator it = first; it != last; ++it) {
    partial = binary_op(partial, *it);
    *(result++) = partial;
  }
  return result;
}
} // namespace emu

template <typename InputContainer, typename OutputContainer,
          class BinaryOperation>
void test(queue q, InputContainer input, OutputContainer output,
          BinaryOperation binary_op,
          typename OutputContainer::value_type identity) {
  typedef typename InputContainer::value_type InputT;
  typedef typename OutputContainer::value_type OutputT;
  typedef class inclusive_scan_kernel<BinaryOperation, 0> kernel_name0;
  typedef class inclusive_scan_kernel<BinaryOperation, 1> kernel_name1;
  typedef class inclusive_scan_kernel<BinaryOperation, 2> kernel_name2;
  typedef class inclusive_scan_kernel<BinaryOperation, 3> kernel_name3;
  OutputT init = 42;
  size_t N = input.size();
  size_t G = 16;
  std::vector<OutputT> expected(N);
  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
      auto in = in_buf.template get_access<access::mode::read>(cgh);
      auto out = out_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for<kernel_name0>(nd_range<1>(G, G), [=](nd_item<1> it) {
        group<1> g = it.get_group();
        int lid = it.get_local_id(0);
        out[lid] = inclusive_scan(g, in[lid], binary_op);
      });
    });
  }
  emu::inclusive_scan(input.begin(), input.begin() + G, expected.begin(),
                      binary_op, identity);
  assert(std::equal(output.begin(), output.begin() + G, expected.begin()));

  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
      auto in = in_buf.template get_access<access::mode::read>(cgh);
      auto out = out_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for<kernel_name1>(nd_range<1>(G, G), [=](nd_item<1> it) {
        group<1> g = it.get_group();
        int lid = it.get_local_id(0);
        out[lid] = inclusive_scan(g, in[lid], binary_op, init);
      });
    });
  }
  emu::inclusive_scan(input.begin(), input.begin() + G, expected.begin(),
                      binary_op, init);
  assert(std::equal(output.begin(), output.begin() + G, expected.begin()));

  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
      auto in = in_buf.template get_access<access::mode::read>(cgh);
      auto out = out_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for<kernel_name2>(nd_range<1>(G, G), [=](nd_item<1> it) {
        group<1> g = it.get_group();
        inclusive_scan(g, in.get_pointer(), in.get_pointer() + N,
                       out.get_pointer(), binary_op);
      });
    });
  }
  emu::inclusive_scan(input.begin(), input.begin() + N, expected.begin(),
                      binary_op, identity);
  assert(std::equal(output.begin(), output.begin() + N, expected.begin()));

  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
      auto in = in_buf.template get_access<access::mode::read>(cgh);
      auto out = out_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for<kernel_name3>(nd_range<1>(G, G), [=](nd_item<1> it) {
        group<1> g = it.get_group();
        inclusive_scan(g, in.get_pointer(), in.get_pointer() + N,
                       out.get_pointer(), binary_op, init);
      });
    });
  }
  emu::inclusive_scan(input.begin(), input.begin() + N, expected.begin(),
                      binary_op, init);
  assert(std::equal(output.begin(), output.begin() + N, expected.begin()));
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
  std::array<int, N> output;
  std::iota(input.begin(), input.end(), 0);
  std::fill(output.begin(), output.end(), 0);

#if __cplusplus >= 201402L
  test(q, input, output, plus<>(), 0);
  test(q, input, output, minimum<>(), std::numeric_limits<int>::max());
  test(q, input, output, maximum<>(), std::numeric_limits<int>::lowest());
#endif
  test(q, input, output, plus<int>(), 0);
  test(q, input, output, minimum<int>(), std::numeric_limits<int>::max());
  test(q, input, output, maximum<int>(), std::numeric_limits<int>::lowest());

  std::cout << "Test passed." << std::endl;
}
