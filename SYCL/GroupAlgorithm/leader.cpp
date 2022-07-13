// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <cassert>
#include <sycl/sycl.hpp>
using namespace sycl;
using namespace sycl::ext::oneapi;

class leader_kernel;

void test(queue q) {
  typedef class leader_kernel kernel_name;
  int out = 0;
  size_t G = 4;

  range<2> R(G, G);
  {
    buffer<int> out_buf(&out, 1);

    q.submit([&](handler &cgh) {
      auto out = out_buf.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<kernel_name>(nd_range<2>(R, R), [=](nd_item<2> it) {
        group<2> g = it.get_group();
        if (leader(g)) {
          out[0] += 1;
        }
      });
    });
  }
  assert(out == 1);
}

int main() {
  queue q;
  std::string version = q.get_device().get_info<info::device::version>();
  if (version < std::string("2.0")) {
    std::cout << "Skipping test\n";
    return 0;
  }

  test(q);

  std::cout << "Test passed." << std::endl;
}
