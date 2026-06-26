#include "a.hpp"

void submitKernelWithIdB(sycl::queue &Q, int *Ptr) {
  enqueueWithKernelId<TestKernel>(Q, Ptr);
}