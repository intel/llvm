// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

int main() {
  ::sycl::queue q;
  int r = 0;
  {
    sycl::buffer<int, 1> b(&r, 1);
    q.submit([&](sycl::handler &h) {
      auto a = b.get_access<sycl::access::mode::write>(h);
      h.single_task<class T>([=]() { a[0] = 42; });
    });
  }
  return r - 42;
}
