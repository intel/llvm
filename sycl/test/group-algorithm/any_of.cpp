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

template <class Predicate>
class any_of_kernel;

struct GeZero {
  bool operator()(int i) const { return i >= 0; }
};
struct IsEven {
  bool operator()(int i) const { return (i % 2) == 0; }
};
struct LtZero {
  bool operator()(int i) const { return i < 0; }
};

template <typename InputContainer, typename OutputContainer, class Predicate>
void test(queue q, InputContainer input, OutputContainer output,
          Predicate pred) {
  typedef typename InputContainer::value_type InputT;
  typedef typename OutputContainer::value_type OutputT;
  typedef class any_of_kernel<Predicate> kernel_name;
  size_t N = input.size();
  size_t G = 16;
  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto in = in_buf.template get_access<access::mode::read>(cgh);
      auto out = out_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for<kernel_name>(nd_range<1>(G, G), [=](nd_item<1> it) {
        group<1> g = it.get_group();
        int lid = it.get_local_id(0);
        out[0] = any_of(g, pred(in[lid]));
        out[1] = any_of(g, in[lid], pred);
        out[2] = any_of(g, in.get_pointer(), in.get_pointer() + N, pred);
      });
    });
  }
  bool expected = std::any_of(input.begin(), input.end(), pred);
  assert(output[0] == expected);
  assert(output[1] == expected);
  assert(output[2] == expected);
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
  std::array<bool, 3> output;
  std::iota(input.begin(), input.end(), 0);
  std::fill(output.begin(), output.end(), false);

  test(q, input, output, GeZero());
  test(q, input, output, IsEven());
  test(q, input, output, LtZero());

  std::cout << "Test passed." << std::endl;
}
