// REQUIRES: arch-intel_gpu_dg2_g10 || arch-intel_gpu_dg2_g11 || arch-intel_gpu_dg2_g12

// RUN: %{build} -fsycl-targets=intel_gpu_dg2 -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

int choose_dg2(const sycl::device& d) {
  auto ver = d.get_info<sycl::info::device::version>();
  return ver == "12.55.8" || ver == "12.56.5" || ver == "12.57.0";
}

int main() {
  sycl::device dev { choose_dg2 };
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
