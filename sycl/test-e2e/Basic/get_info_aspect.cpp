/* This test checks that get_info checks its aspect and passes without ZES_ENABLE_SYSMAN=1.*/
// REQUIRES: gpu-intel-dg2, level_zero
// RUN: %{build} -o %t.out
// RUN: env ZES_ENABLE_SYSMAN=0 %{run} %t.out
// Explicitly set 'ZES_ENABLE_SYSMAN=0'. HWLOC initializes this environment
// variable in its constructor, causing this test to fail, as retrieving
// free memory information is expected not to work in this test.
// For more context, see: https://github.com/oneapi-src/level-zero/issues/36.

#include <sycl/detail/core.hpp>
int main() {
  sycl::queue q;
  bool failed = false;
  try {
    sycl::device d(sycl::default_selector_v);
    size_t mem_free = d.get_info<sycl::ext::intel::info::device::free_memory>();
  } catch (const sycl::exception &e) {
    assert(e.code() == sycl::errc::feature_not_supported);
    std::cout << "UnExpected exception encountered: " << e.what() << std::endl;
    failed = true;
  }
  return failed;
}
