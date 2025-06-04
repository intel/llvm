// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

int main() {
  sycl::queue q;
  sycl::buffer<int, 1> b{1};
  sycl::host_accessor{b}[0] = 0;

  auto l = [&](sycl::handler &cgh) {
    sycl::accessor acc{b, cgh};
    cgh.single_task([=]() { ++acc[0]; });
  };
  q.submit(l);
  assert(sycl::host_accessor{b}[0] == 1);

  const auto cl = [&](sycl::handler &cgh) {
    sycl::accessor acc{b, cgh};
    cgh.single_task([=]() { ++acc[0]; });
  };
  q.submit(cl);
  assert(sycl::host_accessor{b}[0] == 2);

  return 0;
}
