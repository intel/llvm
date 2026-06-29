#include "a.hpp"

void submitKernelWithIdA(sycl::queue &Q, int *Ptr) {
  enqueueWithKernelId(Q, sycl::get_kernel_id<TestKernel>(), Ptr);
}