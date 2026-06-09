// RUN: not %clangxx -fsycl %s 2>&1 | FileCheck %s

#include <sycl/sycl.hpp>

class NotAKernel;

int main() {
  // CHECK: undefined reference to `sycl::_V1::detail::getKernelNameHelper(sycl::_V1::detail::KernelIdentity<NotAKernel>)
  sycl::get_kernel_id<NotAKernel>();
}
