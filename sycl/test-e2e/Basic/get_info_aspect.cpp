/* This test checks that get_info checks its aspect and throws an invalid object
   error when ext::intel::info::device::free_memory is missing on L0*/

// REQUIRES: gpu, level_zero
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
int main() {
  sycl::queue q;
  bool failed = true;
  try {
    sycl::device d(sycl::default_selector_v);
    size_t mem_free = d.get_info<sycl::ext::intel::info::device::free_memory>();
  } catch (const sycl::invalid_object_error &e) {
    std::cout << "Expected exception encountered: " << e.what() << std::endl;
    failed = false;
  }
  return failed;
}
