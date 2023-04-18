// REQUIRES: cuda || hip
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  sycl::device dev = q.get_device();

  auto max_regs_per_wg =
      dev.get_info<sycl::ext::codeplay::experimental::info::device::
                       max_registers_per_work_group>();
  std::cout << "Max register per work-group: " << max_regs_per_wg << std::endl;

  assert(max_regs_per_wg > 0);

  std::cout << "Passed!" << std::endl;
  return 0;
}
