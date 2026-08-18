#include <sycl/detail/core.hpp>

int main() {
  sycl::queue q;

  int data = 0;
  {
    sycl::buffer<int, 1> buf(&data, sycl::range<1>(1));

    q.submit([&](sycl::handler &h) {
       auto acc = buf.get_access<sycl::access::mode::write>(h);
       h.single_task([=]() { acc[0] = 42; });
     }).wait();
  }

  assert(data == 42);

  return 0;
}
