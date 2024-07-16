// RUN: %{build} -fsycl-device-code-split=per_kernel -I . -o %t.out
// RUN: %{run} %t.out

#include "../helpers.hpp"
#include "support.h"
#include <algorithm>
#include <cassert>
#include <complex>
#include <iostream>
#include <limits>
#include <numeric>
#include <sycl/detail/core.hpp>
#include <sycl/group_algorithm.hpp>
#include <vector>
using namespace sycl;

template <class SpecializationKernelName, int TestNumber>
class exclusive_scan_kernel;

queue q;

template <typename SpecializationKernelName, typename InputContainer,
          class BinaryOperation>
void test(const InputContainer &input, BinaryOperation binary_op,
          typename InputContainer::value_type identity) {
  typedef typename InputContainer::value_type InputT;
  typedef InputT OutputT;
  typedef class exclusive_scan_kernel<SpecializationKernelName, 0> kernel_name0;
  constexpr size_t G = 16;
  constexpr size_t N = std::tuple_size_v<InputContainer>; // 128 or 12
  constexpr size_t confirmRange = std::min(G, N);
  std::array<OutputT, N> output;
  std::array<OutputT, N> expected;

  // checking
  // template <typename Group, typename T, typename BinaryOperation>
  // T exclusive_scan_over_group(Group g, T x, BinaryOperation binary_op)
  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for<kernel_name0>(
          nd_range<1>(confirmRange, confirmRange), [=](nd_item<1> it) {
            group<1> g = it.get_group();
            int lid = it.get_local_id(0);
            out[lid] = exclusive_scan_over_group(g, in[lid], binary_op);
          });
    });
  }
  emu::exclusive_scan(input.begin(), input.begin() + confirmRange,
                      expected.begin(), identity, binary_op);
  assert(ranges_equal(output.begin(), output.begin() + confirmRange,
                      expected.begin()));

  typedef class exclusive_scan_kernel<SpecializationKernelName, 1> kernel_name1;
  constexpr OutputT init(42);

  // checking
  // template <typename Group, typename V, typename T, class BinaryOperation>
  // T exclusive_scan_over_group(Group g, V x, T init, BinaryOperation
  // binary_op)
  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for<kernel_name1>(
          nd_range<1>(confirmRange, confirmRange), [=](nd_item<1> it) {
            group<1> g = it.get_group();
            int lid = it.get_local_id(0);
            out[lid] = exclusive_scan_over_group(g, in[lid], init, binary_op);
          });
    });
  }
  emu::exclusive_scan(input.begin(), input.begin() + confirmRange,
                      expected.begin(), init, binary_op);
  assert(ranges_equal(output.begin(), output.begin() + confirmRange,
                      expected.begin()));

  typedef class exclusive_scan_kernel<SpecializationKernelName, 2> kernel_name2;

  // checking
  // template <typename Group, typename InPtr, typename OutPtr,
  //           class BinaryOperation>
  // OutPtr joint_exclusive_scan(Group g, InPtr first, InPtr last, OutPtr
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
        joint_exclusive_scan(
            g, in.template get_multi_ptr<access::decorated::no>(),
            in.template get_multi_ptr<access::decorated::no>() + N,
            out.template get_multi_ptr<access::decorated::no>(), binary_op);
      });
    });
  }
  emu::exclusive_scan(input.begin(), input.begin() + N, expected.begin(),
                      identity, binary_op);
  assert(ranges_equal(output.begin(), output.begin() + N, expected.begin()));

  typedef class exclusive_scan_kernel<SpecializationKernelName, 3> kernel_name3;

  // checking
  // template <typename Group, typename InPtr, typename OutPtr, typename T,
  //      class BinaryOperation>
  // OutPtr joint_exclusive_scan(Group g, InPtr first, InPtr last, OutPtr
  // result, T init,
  //                  BinaryOperation binary_op)
  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for<kernel_name3>(nd_range<1>(G, G), [=](nd_item<1> it) {
        group<1> g = it.get_group();
        joint_exclusive_scan(
            g, in.template get_multi_ptr<access::decorated::no>(),
            in.template get_multi_ptr<access::decorated::no>() + N,
            out.template get_multi_ptr<access::decorated::no>(), init,
            binary_op);
      });
    });
  }
  emu::exclusive_scan(input.begin(), input.begin() + N, expected.begin(), init,
                      binary_op);
  assert(ranges_equal(output.begin(), output.begin() + N, expected.begin()));
}

int main() {
  queue q;
  if (!isSupportedDevice(q.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }

  constexpr int N = 128;
  std::array<int, N> input;
  std::iota(input.begin(), input.end(), 2);

  // Smaller size as the multiplication test
  // will result in computing of a factorial
  // 12! fits in a 32 bits integer.
  constexpr int M = 12;
  std::array<int, M> input_small;
  std::iota(input_small.begin(), input_small.end(), 1);

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
  test<class KernelNameMultipliesI>(input_small, sycl::multiplies<int>(), 1);
  test<class KernelNameBitOrI>(input, sycl::bit_or<int>(), 0);
  test<class KernelNameBitXorI>(input, sycl::bit_xor<int>(), 0);
  test<class KernelNameBitAndI>(input_small, sycl::bit_and<int>(), ~0);

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
  // https://github.com/intel/llvm/pull/5108/ ) joint_exclusive_scan and
  // exclusive_scan_over_group now operate on std::complex but limited to the
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
