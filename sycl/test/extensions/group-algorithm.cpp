// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -I . -o %t.out
// Group operations are not supported on host device. The test checks that
// compilation succeeded.

// TODO: enable compile+runtime checks for operations defined in SPIR-V 1.3.
// That requires either adding a switch to clang (-spirv-max-version=1.3) or
// raising the spirv version from 1.1. to 1.3 for spirv translator
// unconditionally. Using operators specific for spirv 1.3 and higher with
// -spirv-max-version=1.1 being set by default causes assert/check fails
// in spirv translator.
// RUNx: %clangxx -fsycl -fsycl-targets=%sycl_triple -DSPIRV_1_3 %s -I . -o \
   %t13.out

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>
using namespace sycl;
using namespace sycl::ext::oneapi;

template <class Predicate> class none_of_kernel;

struct GeZero {
  bool operator()(int i) const { return i >= 0; }
};
struct IsEven {
  bool operator()(int i) const { return (i % 2) == 0; }
};
struct LtZero {
  bool operator()(int i) const { return i < 0; }
};

template <typename SpecializationKernelName, typename InputContainer,
          typename OutputContainer, class BinaryOperation, class Predicate>
void test(queue q, InputContainer input, OutputContainer output,
          BinaryOperation binary_op,
          typename OutputContainer::value_type identity, Predicate pred) {
  typedef typename InputContainer::value_type InputT;
  typedef typename OutputContainer::value_type OutputT;
  OutputT init = 42;
  size_t N = input.size();
  size_t G = 64;
  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto in = in_buf.template get_access<access::mode::read>(cgh);
      auto out = out_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for<SpecializationKernelName>(
          nd_range<1>(G, G), [=](nd_item<1> it) {
            group<1> g = it.get_group();
            int lid = it.get_local_id(0);
            out[0] = reduce(g, in[lid], binary_op);
            out[1] = none_of(g, in[lid], pred);
            out[2] = inclusive_scan(g, in[lid], binary_op);
            out[3] = exclusive_scan(g, in[lid], binary_op);
            out[4] = broadcast(g, in[lid]);
            out[5] = any_of(g, in.get_pointer(), in.get_pointer() + N, pred);
            out[6] = all_of(g, pred(in[lid]));
            if (leader(g)) {
              out[7]++;
            }
          });
    });
  }
}

int main() {
  queue q;

  constexpr int N = 128;
  std::array<int, N> input;
  std::array<int, 8> output;
  std::iota(input.begin(), input.end(), 0);
  std::fill(output.begin(), output.end(), 0);

  test<class KernelNamePlusV>(q, input, output, ext::oneapi::plus<>(), 0,
                              GeZero());
  test<class KernelNameMinimumV>(q, input, output, ext::oneapi::minimum<>(),
                                 std::numeric_limits<int>::max(), IsEven());

#ifdef SPIRV_1_3
  test<class KernelName_WonwuUVPUPOTKRKIBtT>(
      q, input, output, ext::oneapi::multiplies<int>(), 1, LtZero());
#endif // SPIRV_1_3

  std::cout << "Test passed." << std::endl;
}
