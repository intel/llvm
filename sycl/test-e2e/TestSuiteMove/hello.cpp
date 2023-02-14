// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// XFAIL: cpu

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  return q.get_device().get_info<sycl::info::device::device_type>() ==
         sycl::info::device_type::cpu;
}
