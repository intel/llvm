// RUN: %{build} -fsycl-embed-ir -I . -o %t.out
// RUN: %{run} %t.out

#include "../../helpers.hpp"
#include "../helpers.hpp"
#include "support.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <numeric>
#include <sycl/detail/core.hpp>
#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>
#include <sycl/properties/all_properties.hpp>
#include <vector>
using namespace sycl;

// COM: Check exclusive_scan works with fusion

template <class SpecializationKernelName, int TestNumber>
class exclusive_scan_kernel;

template <typename BinaryOperation> class K0;
template <typename BinaryOperation> class K1;
template <typename BinaryOperation> class K2;
template <typename BinaryOperation> class K3;

template <typename InputContainer, typename OutputContainer,
          class BinaryOperation>
void test(queue q, InputContainer input, OutputContainer output,
          BinaryOperation binary_op,
          typename OutputContainer::value_type identity) {
  typedef typename InputContainer::value_type InputT;
  typedef typename OutputContainer::value_type OutputT;
  OutputT init = 42;
  size_t N = input.size();
  size_t G = 64;
  std::vector<OutputT> expected(N);
  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());
    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();

    iota(q, in_buf, 0);

    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for<K0<BinaryOperation>>(
          nd_range<1>(G, G), [=](nd_item<1> it) {
            group<1> g = it.get_group();
            int lid = it.get_local_id(0);
            out[lid] = exclusive_scan_over_group(g, in[lid], binary_op);
          });
    });

    complete_fusion_with_check(
        fw, ext::codeplay::experimental::property::no_barriers{});
  }
  emu::exclusive_scan(input.begin(), input.begin() + G, expected.begin(),
                      identity, binary_op);
  assert(std::equal(output.begin(), output.begin() + G, expected.begin()));

  // Fill to test fusion again
  std::fill(input.begin(), input.end(), 0);

  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());

    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();

    iota(q, in_buf, 0);

    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for<K1<BinaryOperation>>(
          nd_range<1>(G, G), [=](nd_item<1> it) {
            group<1> g = it.get_group();
            int lid = it.get_local_id(0);
            out[lid] = exclusive_scan_over_group(g, in[lid], init, binary_op);
          });
    });

    complete_fusion_with_check(
        fw, ext::codeplay::experimental::property::no_barriers{});
  }
  emu::exclusive_scan(input.begin(), input.begin() + G, expected.begin(), init,
                      binary_op);
  assert(std::equal(output.begin(), output.begin() + G, expected.begin()));

  // Fill to test fusion again
  std::fill(input.begin(), input.end(), 0);

  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());

    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();

    iota(q, in_buf, 0);

    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for<K2<BinaryOperation>>(
          nd_range<1>(G, G), [=](nd_item<1> it) {
            group<1> g = it.get_group();
            joint_exclusive_scan(
                g, in.template get_multi_ptr<access::decorated::no>(),
                in.template get_multi_ptr<access::decorated::no>() + N,
                out.template get_multi_ptr<access::decorated::no>(), binary_op);
          });
    });
    complete_fusion_with_check(
        fw, ext::codeplay::experimental::property::no_barriers{});
  }
  emu::exclusive_scan(input.begin(), input.begin() + N, expected.begin(),
                      identity, binary_op);
  assert(std::equal(output.begin(), output.begin() + N, expected.begin()));

  // Fill to test fusion again
  std::fill(input.begin(), input.end(), 0);

  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());
    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();

    iota(q, in_buf, 0);

    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for<K3<BinaryOperation>>(
          nd_range<1>(G, G), [=](nd_item<1> it) {
            group<1> g = it.get_group();
            joint_exclusive_scan(
                g, in.template get_multi_ptr<access::decorated::no>(),
                in.template get_multi_ptr<access::decorated::no>() + N,
                out.template get_multi_ptr<access::decorated::no>(), init,
                binary_op);
          });
    });
    complete_fusion_with_check(
        fw, ext::codeplay::experimental::property::no_barriers{});
  }
  emu::exclusive_scan(input.begin(), input.begin() + N, expected.begin(), init,
                      binary_op);
  assert(std::equal(output.begin(), output.begin() + N, expected.begin()));
}

int main() {
  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};
  if (!isSupportedDevice(q.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }

  constexpr int N = 128;
  std::array<int, N> input;
  std::array<int, N> output;
  std::fill(output.begin(), output.end(), 0);

  test(q, input, output, sycl::plus<>(), 0);
  test(q, input, output, sycl::minimum<>(), std::numeric_limits<int>::max());
  test(q, input, output, sycl::maximum<>(), std::numeric_limits<int>::lowest());
  test(q, input, output, sycl::plus<int>(), 0);
  test(q, input, output, sycl::minimum<int>(), std::numeric_limits<int>::max());
  test(q, input, output, sycl::maximum<int>(),
       std::numeric_limits<int>::lowest());
  test(q, input, output, sycl::multiplies<int>(), 1);
  test(q, input, output, sycl::bit_or<int>(), 0);
  test(q, input, output, sycl::bit_xor<int>(), 0);
  test(q, input, output, sycl::bit_and<int>(), ~0);

  std::cout << "Test passed." << std::endl;
}
