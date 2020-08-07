// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device %s -emit-llvm -triple spir64-unknown-unknown-sycldevice -o - | FileCheck %s
#include <sycl.hpp>

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {
  cl::sycl::accessor<char, 1, cl::sycl::access::mode::read> Acc1;
  cl::sycl::accessor<char, 1, cl::sycl::access::mode::read> Acc2;
  kernel<class kernel_restrict>(
      [=]() [[intel::kernel_args_restrict]] { Acc1.use();Acc2.use(); });
  // CHECK: define spir_kernel {{.*}}kernel_restrict(i8 addrspace(1)* noalias %{{.*}}, i8 addrspace(1)* noalias %{{.*}})

  kernel<class kernel_norestrict>(
      [=]() { Acc1.use();Acc2.use(); });
  // CHECK: define spir_kernel {{.*}}kernel_norestrict(i8 addrspace(1)* %{{.*}}, i8 addrspace(1)* %{{.*}})

  int g = 42;
  kernel<class kernel_restrict_other_types>(
      [=]() [[intel::kernel_args_restrict]] { Acc1.use();Acc2.use();
      int a = g; });
  // CHECK: define spir_kernel {{.*}}kernel_restrict_other_types(i8 addrspace(1)* noalias %{{.*}}, i8 addrspace(1)* noalias %{{.*}}, i32 %{{.*}})
}
