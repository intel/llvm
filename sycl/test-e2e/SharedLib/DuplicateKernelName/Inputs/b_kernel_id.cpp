#include "a.hpp"

void submitKernelWithIdB(sycl::queue &Q, int *Ptr) {
  enqueueWithKernelId(Q, sycl::get_kernel_id<TestKernel>(), Ptr);
}
