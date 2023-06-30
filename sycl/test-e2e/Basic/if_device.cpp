// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/sycl.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue q;
  sycl::buffer<int, 1> b(1);

  {
    sycl::host_accessor a{b};
    a[0] = 1;
    syclex::if_device([&]() { a[0] = 2; });
    assert(a[0] == 1);
    syclex::if_device([&]() { a[0] = 2; }).otherwise([&]() { a[0] = 3; });
    assert(a[0] == 3);
    syclex::if_host([&]() { a[0] = 2; });
    assert(a[0] == 2);
    syclex::if_host([&]() { a[0] = 1; }).otherwise([&]() { a[0] = 3; });
    assert(a[0] == 1);
  }
  auto Do = [&](auto Fn) {
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor a{b, cgh};
      cgh.single_task([=]() { Fn(a); });
    });
  };

  Do([&](auto a) { syclex::if_device([&]() { a[0] = 2; }); });
  assert(sycl::host_accessor{b}[0] == 2);
  Do([&](auto a) {
    syclex::if_device([&]() { a[0] = 3; }).otherwise([&]() { a[0] = 1; });
  });
  assert(sycl::host_accessor{b}[0] == 3);
  Do([&](auto a) { syclex::if_host([&]() { a[0] = 2; }); });
  assert(sycl::host_accessor{b}[0] == 3);
  Do([&](auto a) {
    syclex::if_host([&]() { a[0] = 2; }).otherwise([&]() { a[0] = 1; });
  });
  assert(sycl::host_accessor{b}[0] == 1);

  return 0;
}
