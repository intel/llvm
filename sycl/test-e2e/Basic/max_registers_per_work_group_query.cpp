// REQUIRES: cuda || hip
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

int main() {
  sycl::queue q;
  sycl::device dev = q.get_device();

#if !defined(SYCL_EXT_CODEPLAY_MAX_REGISTERS_PER_WORK_GROUP_QUERY)
#error SYCL_EXT_CODEPLAY_MAX_REGISTERS_PER_WORK_GROUP_QUERY is not defined!
#endif

  auto max_regs_per_wg =
      dev.get_info<sycl::ext::codeplay::experimental::info::device::
                       max_registers_per_work_group>();
  std::cout << "Max register per work-group: " << max_regs_per_wg << std::endl;

  assert(max_regs_per_wg > 0);

  std::cout << "Passed!" << std::endl;
  return 0;
}
