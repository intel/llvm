// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

using namespace cl::sycl;

void test_conversion(queue q) {
  int init = 0;
  {
    buffer<int> in_buf(&init, 1);

    q.submit([&](handler &cgh) {
      auto in = in_buf.template get_access<access::mode::atomic>(cgh);
      cgh.single_task<class conversion>([=]() {
        cl::sycl::atomic<int, access::address_space::global_space> atm = in[0];
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
