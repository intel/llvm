// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump -sycl-std=2020 %s | FileCheck %s

// This test checks that the compiler generates correct kernel wrapper arguments for
// different accessors targets.

#include "sycl.hpp"

sycl::queue q;

int main() {
  // Access work-group local memory with read and write access.
  sycl::accessor<int, 1, sycl::access::mode::read_write,
                 sycl::access::target::local>
      local_acc;
  // Access buffer via global memory with read and write access.
  sycl::accessor<int, 1, sycl::access::mode::read_write,
                 sycl::access::target::global_buffer>
      global_acc;
  // Access buffer via constant memory with read and write access.
  sycl::accessor<int, 1, sycl::access::mode::read_write,
                 sycl::access::target::constant_buffer>
      constant_acc;

  q.submit([&](sycl::handler &h) {
    h.single_task<class use_local>(
        [=] {
          local_acc.use();
        });
  });

  q.submit([&](sycl::handler &h) {
    h.single_task<class use_global>(
        [=] {
          global_acc.use();
        });
  });

  q.submit([&](sycl::handler &h) {
    h.single_task<class use_constant>(
        [=] {
          constant_acc.use();
        });
  });
}
// CHECK: {{.*}}use_local{{.*}} 'void (__local int *, sycl::range<1>, sycl::range<1>, sycl::id<1>)'
// CHECK: {{.*}}use_global{{.*}} 'void (__global int *, sycl::range<1>, sycl::range<1>, sycl::id<1>)'
// CHECK: {{.*}}use_constant{{.*}} 'void (__constant int *, sycl::range<1>, sycl::range<1>, sycl::id<1>)'
