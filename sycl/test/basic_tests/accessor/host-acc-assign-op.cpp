// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

int main() {
  using AccT = sycl::host_accessor<int, 0>;

  int Expected = 64;
  int Data = 32;
  sycl::buffer<int, 1> DataBuffer(&Data, sycl::range<1>(1));
  AccT HostAcc{DataBuffer};

  HostAcc = Expected;

  assert(HostAcc == Expected);
}
