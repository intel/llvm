// UNSUPPORTED: hip
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-device-code-split=per_kernel %s -I . -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include "support.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <numeric>
#include <sycl/sycl.hpp>
#include <vector>
using namespace sycl;

template <class SpecializationKernelName, int TestNumber>
class inclusive_scan_kernel;

// std::inclusive_scan isn't implemented yet, so use serial implementation
// instead
// TODO: use std::inclusive_scan when it will be supported
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

template <typename SpecializationKernelName, typename InputContainer,
          typename OutputContainer, class BinaryOperation>
void test(queue q, InputContainer input, OutputContainer output,
          BinaryOperation binary_op,
          typename OutputContainer::value_type identity) {
  typedef typename InputContainer::value_type InputT;
  typedef typename OutputContainer::value_type OutputT;
  typedef class inclusive_scan_kernel<SpecializationKernelName, 0> kernel_name0;
  constexpr size_t N = input.size(); // 128 or 12
  constexpr size_t G = 64;
  constexpr size_t confirmRange = std::min(G, N);
  std::vector<OutputT> expected(N);

  // checking
  // template <typename Group, typename T, class BinaryOperation>
  // T inclusive_scan_over_group(Group g, T x, BinaryOperation binary_op)
  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for<kernel_name0>(nd_range<1>(G, G), [=](nd_item<1> it) {
        group<1> g = it.get_group();
        int lid = it.get_local_id(0);
        out[lid] = inclusive_scan_over_group(g, in[lid], binary_op);
      });
    });
  }
  emu::inclusive_scan(input.begin(), input.begin() + confirmRange,
                      expected.begin(), binary_op, identity);
  assert(std::equal(output.begin(), output.begin() + confirmRange,
                    expected.begin()));

  typedef class inclusive_scan_kernel<SpecializationKernelName, 1> kernel_name1;
  constexpr OutputT init = 42;

  // checking
  // template <typename Group, typename V, class BinaryOperation, typename T>
  // T inclusive_scan_over_group(Group g, V x, BinaryOperation binary_op, T
  // init)
  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for<kernel_name1>(nd_range<1>(G, G), [=](nd_item<1> it) {
        group<1> g = it.get_group();
        int lid = it.get_local_id(0);
        out[lid] = inclusive_scan_over_group(g, in[lid], binary_op, init);
      });
    });
  }
  emu::inclusive_scan(input.begin(), input.begin() + confirmRange,
                      expected.begin(), binary_op, init);
  assert(std::equal(output.begin(), output.begin() + confirmRange,
                    expected.begin()));

  typedef class inclusive_scan_kernel<SpecializationKernelName, 2> kernel_name2;

  // checking
  // template <typename Group, typename InPtr, typename OutPtr,
  //           class BinaryOperation>
  // OutPtr joint_inclusive_scan(Group g, InPtr first, InPtr last, OutPtr
  // result,
  //                  BinaryOperation binary_op)
  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for<kernel_name2>(nd_range<1>(G, G), [=](nd_item<1> it) {
        group<1> g = it.get_group();
        joint_inclusive_scan(g, in.get_pointer(), in.get_pointer() + N,
                             out.get_pointer(), binary_op);
      });
    });
  }
  emu::inclusive_scan(input.begin(), input.begin() + N, expected.begin(),
                      binary_op, identity);
  assert(std::equal(output.begin(), output.begin() + N, expected.begin()));

  typedef class inclusive_scan_kernel<SpecializationKernelName, 3> kernel_name3;

  // checking
  // template <typename Group, typename InPtr, typename OutPtr,
  //      class BinaryOperation, typename T>
  // OutPtr joint_inclusive_scan(Group g, InPtr first, InPtr last, OutPtr
  // result,
  //                             BinaryOperation binary_op, T init)
  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for<kernel_name3>(nd_range<1>(G, G), [=](nd_item<1> it) {
        group<1> g = it.get_group();
        joint_inclusive_scan(g, in.get_pointer(), in.get_pointer() + N,
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
  if (!isSupportedDevice(q.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }

  constexpr int N = 128;
  std::array<int, N> input;
  std::array<int, N> output;
  std::iota(input.begin(), input.end(), 2);
  std::fill(output.begin(), output.end(), 0);

  // Smaller size as the multiplication test
  // will result in computing of a factorial
  // 12! fits in a 32 bits integer.
  constexpr int M = 12;
  std::array<int, M> input_small;
  std::array<int, M> output_small;
  std::iota(input_small.begin(), input_small.end(), 1);
  std::fill(output_small.begin(), output_small.end(), 0);

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
  test<class KernelNameMultipliesI>(q, input_small, output_small,
                                    sycl::multiplies<int>(), 1);
  test<class KernelNameBitOrI>(q, input, output, sycl::bit_or<int>(), 0);
  test<class KernelNameBitXorI>(q, input, output, sycl::bit_xor<int>(), 0);
  test<class KernelNameBitAndI>(q, input_small, output_small,
                                sycl::bit_and<int>(), ~0);

  // as part of SYCL_EXT_ONEAPI_COMPLEX_ALGORITHMS (
  // https://github.com/intel/llvm/pull/5108/ ) joint_inclusive_scan and
  // inclusive_scan_over_group now operate on std::complex limited to using the
  // sycl::plus binary operation.
#ifdef SYCL_EXT_ONEAPI_COMPLEX_ALGORITHMS
  std::array<std::complex<float>, N> input_cf;
  std::array<std::complex<float>, N> output_cf;
  std::iota(input_cf.begin(), input_cf.end(), 0);
  std::fill(output_cf.begin(), output_cf.end(), 0);
  test<class KernelNamePlusComplexF>(q, input_cf, output_cf,
                                     sycl::plus<std::complex<float>>(), 0);
  test<class KernelNamePlusUnspecF>(q, input_cf, output_cf, sycl::plus<>(), 0);

  if (q.get_device().has(aspect::fp64)) {
    std::array<std::complex<double>, N> input_cd;
    std::array<std::complex<double>, N> output_cd;
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
