// UNSUPPORTED: cuda
// OpenCL C 2.x alike work-group functions not yet supported by CUDA.
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// TODO: enable compile+runtime checks for operations defined in SPIR-V 1.3.
// That requires either adding a switch to clang (-spirv-max-version=1.3) or
// raising the spirv version from 1.1. to 1.3 for spirv translator
// unconditionally. Using operators specific for spirv 1.3 and higher with
// -spirv-max-version=1.1 being set by default causes assert/check fails
// in spirv translator.
// RUNx: %clangxx -fsycl -fsycl-targets=%sycl_triple -DSPIRV_1_3 %s -o %t13.out

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>
using namespace sycl;
using namespace sycl::ONEAPI;

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
      cgh.parallel_for<SpecializationKernelName>(
          nd_range<1>(G, G), [=](nd_item<1> it) {
            group<1> g = it.get_group();
            int lid = it.get_local_id(0);
            out[0] = reduce(g, in[lid], binary_op);
            out[1] = reduce(g, in[lid], init, binary_op);
            out[2] =
                reduce(g, in.get_pointer(), in.get_pointer() + N, binary_op);
            out[3] = reduce(g, in.get_pointer(), in.get_pointer() + N, init,
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

bool isSupportedDevice(device D) {
  std::string PlatformName = D.get_platform().get_info<info::platform::name>();
  if (PlatformName.find("Level-Zero") != std::string::npos)
    return true;

  if (PlatformName.find("OpenCL") != std::string::npos) {
    std::string Version = D.get_info<info::device::version>();
    size_t Offset = Version.find("OpenCL");
    if (Offset == std::string::npos)
      return false;
    Version = Version.substr(Offset + 7, 3);
    if (Version >= std::string("2.0"))
      return true;
  }

  return false;
}

int main() {
  queue q;
  if (!isSupportedDevice(q.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }

  constexpr int N = 32;
  std::array<int, N> input;
  std::array<int, 4> output;
  std::iota(input.begin(), input.end(), 0);
  std::fill(output.begin(), output.end(), 0);

  test<class KernelNamePlusV>(q, input, output, plus<>(), 0);
  test<class KernelNameMinimumV>(q, input, output, minimum<>(),
                                 std::numeric_limits<int>::max());
  test<class KernelNameMaximumV>(q, input, output, maximum<>(),
                                 std::numeric_limits<int>::lowest());

  test<class KernelNamePlusI>(q, input, output, plus<int>(), 0);
  test<class KernelNameMinimumI>(q, input, output, minimum<int>(),
                                 std::numeric_limits<int>::max());
  test<class KernelNameMaximumI>(q, input, output, maximum<int>(),
                                 std::numeric_limits<int>::lowest());

#ifdef SPIRV_1_3
  test<class KernelName_WonwuUVPUPOTKRKIBtT>(q, input, output,
                                             multiplies<int>(), 1);
  test<class KernelName_qYBaJDZTMGkdIwD>(q, input, output, bit_or<int>(), 0);
  test<class KernelName_eLSFt>(q, input, output, bit_xor<int>(), 0);
  test<class KernelName_uFhJnxSVhNAiFPTG>(q, input, output, bit_and<int>(), ~0);
#endif // SPIRV_1_3

  std::cout << "Test passed." << std::endl;
}
