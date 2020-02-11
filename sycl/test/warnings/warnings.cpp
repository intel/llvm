// RUN: %clangxx -Wall -Wpessimizing-move -Wunused-variable -Wmismatched-tags -Wunneeded-internal-declaration -Werror -fsycl %s -o %t.out

#include <CL/sycl.hpp>

using namespace cl::sycl;

int main(void) {
  // add a very simple kernel to see if compilation succeeds with -Werror
  int data1[10] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

  buffer<int, 1> B(data1, range<1>(10), {property::buffer::use_host_ptr()});
  queue Q;
  Q.submit([&](handler &CGH) {
    auto Accessor = B.get_access<access::mode::read_write>(CGH);
    CGH.parallel_for<class TheSimpleKernel>(range<1>{10}, [=](id<1> Item) {
      Accessor[Item] = 0;
    });
  });

  return 0;
}
