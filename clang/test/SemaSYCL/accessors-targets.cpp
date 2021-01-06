// RUN: %clang_cc1 -fsycl -fsycl-is-device -ast-dump -sycl-std=2020 %s | FileCheck %s

// This test checks that the compiler generates correct kernel wrapper arguments for
// different accessors targets.

#include "Inputs/sycl.hpp"

cl::sycl::queue q;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  // Access work-group local memory with read and write access.
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::local>
      local_acc;
  // Access buffer via global memory with read and write access.
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer>
      global_acc;
  // Access buffer via constant memory with read and write access.
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::constant_buffer>
      constant_acc;

  q.submit([&](cl::sycl::handler &h) {
    h.single_task<class use_local>(
        [=] {
          local_acc.use();
        });
  });

  q.submit([&](cl::sycl::handler &h) {
    h.single_task<class use_global>(
        [=] {
          global_acc.use();
        });
  });

  q.submit([&](cl::sycl::handler &h) {
    h.single_task<class use_constant>(
        [=] {
          constant_acc.use();
        });
  });
}
// CHECK: {{.*}}use_local{{.*}} 'void (__local int *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>)'
// CHECK: {{.*}}use_global{{.*}} 'void (__global int *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>)'
// CHECK: {{.*}}use_constant{{.*}} 'void (__constant int *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>)'
