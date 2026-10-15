// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -triple nvptx64-nvidia-cuda -verify=sycl %s
// RUN: %clang_cc1 -internal-isystem %S/Inputs -verify=cxx %s

#include "sycl.hpp"

int main() {
  sycl::queue Q{};
  Q.submit([&](sycl::handler &cgh) {
    // sycl-warning@+3 {{unknown attribute 'intel::max_work_group_size' ignored; it was removed, use the 'max_work_group_size' kernel property from sycl_ext_oneapi_kernel_properties instead}}
    // cxx-warning@+2 {{unknown attribute 'intel::max_work_group_size' ignored}}
    cgh.single_task<class T>(
        [=]() [[intel::max_work_group_size(1, 1, 1024)]] { volatile int A = 42; });
  });
  return 0;
}
