// RUN: not %clangxx -fsycl %s 2>&1 | FileCheck %s --check-prefix %if windows %{CHECK-WINDOWS%} %else %{CHECK-LINUX%}

#include <sycl/sycl.hpp>

class NotAKernel;

int main() {
  // CHECK-LINUX: {{undefined reference to|undefined symbol:}} {{.?}}sycl::_V1::detail::getKernelNameHelper(sycl::_V1::detail::KernelIdentity<NotAKernel>)
  // CHECK-WINDOWS: unresolved external symbol "char const * __cdecl sycl::_V1::detail::getKernelNameHelper(struct sycl::_V1::detail::KernelIdentity<class NotAKernel>)"
  sycl::get_kernel_id<NotAKernel>();
}
