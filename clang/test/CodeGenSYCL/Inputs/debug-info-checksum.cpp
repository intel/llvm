#include "Inputs/sycl.hpp"

int main() {
  sycl::sampler Sampler;
  sycl::kernel_single_task<class use_kernel_for_test>([=]() {
    Sampler.use();
  });
  return 0;
}
