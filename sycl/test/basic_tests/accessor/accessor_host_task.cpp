// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/sycl.hpp>

int main() {
  using AccT = sycl::accessor<int, 0, sycl::access::mode::read_write,
                              sycl::access::target::host_task>;
  int data(5);
  sycl::range<1> r(1);
  sycl::buffer<int, 1> data_buf(&data, r);
  sycl::queue q;
  q.submit([&](sycl::handler &cgh) { AccT acc(data_buf, cgh); });
  return 0;
}
