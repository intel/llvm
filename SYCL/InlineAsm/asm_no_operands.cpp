// UNSUPPORTED: cuda
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "include/asmhelper.h"
#include <CL/sycl.hpp>
class no_operands_kernel;

int main() {
  // Creating SYCL queue
  cl::sycl::queue Queue;
  cl::sycl::device Device = Queue.get_device();

  if (!isInlineASMSupported(Device) ||
      !Device.has_extension("cl_intel_required_subgroup_size")) {
    std::cout << "Skipping test\n";
    return 0;
  }
  // Size of index space for kernel
  cl::sycl::range<1> NumOfWorkItems{16};

  // Submitting command group(work) to queue
  Queue.submit([&](cl::sycl::handler &cgh) {
    // Executing kernel
    cgh.parallel_for<no_operands_kernel>(
        NumOfWorkItems, [=](cl::sycl::id<1> WIid)
                            [[intel::reqd_sub_group_size(8)]] {
#if defined(__SYCL_DEVICE_ONLY__)
                              asm("barrier");
#endif
                            });
  });
}
