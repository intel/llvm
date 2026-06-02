// REQUIRES: arch-intel_gpu_ptl_u || arch-intel_gpu_ptl_h

// RUN: %{build} -fsycl-targets=intel_gpu_ptl -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

int choose_ptl(const sycl::device& d) {
  auto ver = d.get_info<sycl::info::device::version>();
  return ver == "30.0.4" || ver == "30.1.0" || ver == "30.3.1" ||
         ver == "30.4.4" || ver == "30.5.4";
}

int main() {
  sycl::device dev { choose_ptl };
  sycl::queue q(dev);

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
