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
#include <numeric>
using namespace sycl;
using namespace sycl::intel;

class broadcast_kernel;

template <typename InputContainer, typename OutputContainer>
void test(queue q, InputContainer input, OutputContainer output) {
  typedef typename InputContainer::value_type InputT;
  typedef typename OutputContainer::value_type OutputT;
  typedef class broadcast_kernel kernel_name;
  size_t N = input.size();
  size_t G = 4;
  range<2> R(G, G);
  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto in = in_buf.template get_access<access::mode::read>(cgh);
      auto out = out_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for<kernel_name>(nd_range<2>(R, R), [=](nd_item<2> it) {
        group<2> g = it.get_group();
        int lid = it.get_local_linear_id();
        out[0] = broadcast(g, in[lid]);
        out[1] = broadcast(g, in[lid], group<2>::id_type(1, 2));
        out[2] = broadcast(g, in[lid], group<2>::linear_id_type(2 * G + 1));
      });
    });
  }
  assert(output[0] == input[0]);
  assert(output[1] == input[1 * G + 2]);
  assert(output[2] == input[2 * G + 1]);
}

int main() {
  queue q;
  std::string version = q.get_device().get_info<info::device::version>();
  if (version < std::string("2.0")) {
    std::cout << "Skipping test\n";
    return 0;
  }

  constexpr int N = 16;
  std::array<int, N> input;
  std::array<int, N> output;
  std::iota(input.begin(), input.end(), 1);
  std::fill(output.begin(), output.end(), false);

  test(q, input, output);

  std::cout << "Test passed." << std::endl;
}
