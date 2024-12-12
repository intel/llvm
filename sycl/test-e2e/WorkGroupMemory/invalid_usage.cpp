// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
#include <cassert>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/work_group_memory.hpp>
using namespace sycl::ext::oneapi::experimental;

// As per the spec, a work_group_memory object cannot be used in a single task
// kernel or in a sycl::range kernel. An exception with error code
// errc::kernel_argument must be thrown in such cases. This test verifies this.

int main() {

  sycl::queue q;
  try {
    q.submit([&](sycl::handler &cgh) {
      work_group_memory<int> mem{cgh};
      cgh.single_task([=]() { mem = 42; });
    });
    assert(false && "Work group memory was used in a single_task kernel and an "
                    "exception was not seen"); // Fail, exception was not seen
  } catch (sycl::exception &e) {
    // Exception seen but must verify that the error code is correct
    assert(e.code() == sycl::errc::kernel_argument);
  }
  // Same thing but with a range kernel
  try {
    q.submit([&](sycl::handler &cgh) {
      work_group_memory<int> mem{cgh};
      cgh.parallel_for(sycl::range{1}, [=](sycl::id<> it) { mem = 42; });
    });
    assert(false && "Work group memory was used in a range kernel and an "
                    "exception was not seen"); // Fail, exception was not seen
  } catch (sycl::exception &e) {
    // Exception seen but must verify that the error code is correct
    assert(e.code() == sycl::errc::kernel_argument);
  }
  return 0;
}
