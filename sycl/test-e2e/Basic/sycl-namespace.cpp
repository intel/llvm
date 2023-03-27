// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

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
