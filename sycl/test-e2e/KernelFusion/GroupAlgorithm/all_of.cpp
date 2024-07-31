// RUN: %{build} %{embed-ir} -I . -o %t.out
// RUN: %{run} %t.out

#include "../helpers.hpp"
#include "support.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <sycl/detail/core.hpp>
#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/properties/all_properties.hpp>

// COM: Check all_of works with kernel fusion.

using namespace sycl;

template <class Predicate> class all_of_kernel;

struct IsEven {
  bool operator()(int i) const { return (i % 2) == 0; }
};

template <typename InputContainer, typename OutputContainer, class Predicate>
void test(queue q, InputContainer input, OutputContainer output,
          Predicate pred) {
  typedef class all_of_kernel<Predicate> kernel_name;
  size_t N = input.size();
  size_t G = 64;
  {
    buffer<int> in_buf(input.data(), input.size());
    buffer<bool> out_buf(output.data(), output.size());

    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();

    iota(q, in_buf, 0);

    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for<kernel_name>(nd_range<1>(G, G), [=](nd_item<1> it) {
        group<1> g = it.get_group();
        int lid = it.get_local_id(0);
        out[0] = all_of_group(g, pred(in[lid]));
        out[1] = all_of_group(g, in[lid], pred);
        out[2] = joint_all_of(
            g, in.template get_multi_ptr<access::decorated::no>(),
            in.template get_multi_ptr<access::decorated::no>() + N, pred);
      });
    });

    complete_fusion_with_check(
        fw, ext::codeplay::experimental::property::no_barriers{});
  }
  bool expected = std::all_of(input.begin(), input.end(), pred);
  assert(output[0] == expected);
  assert(output[1] == expected);
  assert(output[2] == expected);
}

int main() {
  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};
  if (!isSupportedDevice(q.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }

  constexpr int N = 128;
  std::array<int, N> input;
  std::array<bool, 3> output;
  std::fill(output.begin(), output.end(), false);

  test(q, input, output, IsEven());

  std::cout << "Test passed." << std::endl;
}
