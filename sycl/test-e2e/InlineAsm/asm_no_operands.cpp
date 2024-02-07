// UNSUPPORTED: cuda, hip
// REQUIRES: gpu,linux,sg-16
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "include/asmhelper.h"
#include <sycl/sycl.hpp>
class no_operands_kernel;

int main() {
  // Creating SYCL queue
  sycl::queue Queue;
  sycl::device Device = Queue.get_device();

  if (!isInlineASMSupported(Device)) {
    std::cout << "Skipping test\n";
    return 0;
  }
  // Size of index space for kernel
  sycl::range<1> NumOfWorkItems{16};

  // Submitting command group(work) to queue
  Queue.submit([&](sycl::handler &cgh) {
    // Executing kernel
    cgh.parallel_for<no_operands_kernel>(
        NumOfWorkItems,
        [=](sycl::id<1> WIid) [[sycl::reqd_sub_group_size(16)]] {
#if defined(__SYCL_DEVICE_ONLY__)
          asm("barrier");
#endif
        });
  });
}
