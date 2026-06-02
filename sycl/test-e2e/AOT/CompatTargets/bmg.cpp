// REQUIRES: arch-intel_gpu_bmg_g21 || arch-intel_gpu_bmg_g31 || arch-intel_gpu_lnl_m

// RUN: %{build} -fsycl-targets=intel_gpu_bmg -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

int choose_bmg(const sycl::device& d) {
  auto ver = d.get_info<sycl::info::device::version>();
  return ver == "20.4.4" || ver == "20.2.0" || ver == "20.1.0";
}

int main() {
  sycl::device dev { choose_bmg };
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
