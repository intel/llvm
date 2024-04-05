// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

using namespace sycl;

void test_conversion(queue q) {
  int init = 0;
  {
    buffer<int> in_buf(&init, 1);

    q.submit([&](handler &cgh) {
      auto in = in_buf.template get_access<access::mode::atomic>(cgh);
      cgh.single_task<class conversion>([=]() {
        sycl::atomic<int, access::address_space::global_space> atm = in[0];
        atm.store(42);
      });
    });
  }
  assert(init == 42 && "verification failed");
}

int main() {
  queue q;
  test_conversion(q);
}
