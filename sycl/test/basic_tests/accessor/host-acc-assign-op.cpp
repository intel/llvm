// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

int main() {
  int Expected = 64;
  int Data = 32;
  sycl::buffer<int, 1> DataBuffer(&Data, sycl::range<1>(1));
  sycl::host_accessor<int, 0> HostAcc{DataBuffer};

  HostAcc = Expected;

  assert(HostAcc == Expected);
}
