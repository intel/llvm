#include "a.hpp"

void submitKernelWithIdA(sycl::queue &Q, int *Ptr) {
  enqueueWithKernelId<TestKernel>(Q, Ptr);
}