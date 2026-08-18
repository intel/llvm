#include "b.hpp"

void submitKernelDirectB(sycl::queue &Q, int *Ptr) {
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task<TestKernel>([=]() { Ptr[0] = 2; });
  });
}
