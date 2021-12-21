#include "Inputs/sycl.hpp"

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  cl::sycl::sampler Sampler;
  kernel<class use_kernel_for_test>([=]() {
    Sampler.use();
  });
  return 0;
}
